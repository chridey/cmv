import subprocess
import tempfile
import collections
import os

import numpy as np
import scipy

binDir = os.path.split(__file__)[0]

learnCommandTemplate = os.path.join(binDir, 'svm_sle_learn -v 3 {} {} {} {}')
#svm_sle_learn -v [verbosity] [params] [data_file] [latent_file] [model_file]

classifyCommandTemplate = os.path.join(binDir, 'svm_sle_classify {} {} {} {}')
#svm_sle_classify ([-l]) [data_file] [model_file] [output_file]

def write_params(C, extraction_size, feature_mode, norm_mode, window_mode):
    param_string = '-c {} -l {} -m {} -n {}'.format(C, extraction_size,
                                                    feature_mode, norm_mode)
    if window_mode:
        return param_string + ' -i'

    return param_string

def generate_hidden_variables(h_init, X_s, extraction_size):
    #h_init is how to initialize the hidden variables, or a list of lists
    #if None, use all the data
    #if 'random', randomly select extraction_size percent
    #if 'last' or 'first', select first or last extraction_size percent

    if h_init is None:
        return [range(x.shape[0]) for x in X_s]
    elif h_init == 'random':
        return [sorted(np.random.choice(x.shape[0],
                                        int(extraction_size*x.shape[0]),
                                        False)) for x in X_s]
    elif h_init == 'last':
        return [range(x.shape[0])[-int(extraction_size*x.shape[0]):] for x in X_s]
    elif h_init == 'first':
        return [range(x.shape[0])[:int(extraction_size*x.shape[0])] for x in X_s]
    elif type(h_init) == str:
        raise Exception

    return [h_init_s[:int(extraction_size*len(h_init_s)+1)] for h_init_s in h_init]

def write_hidden_variables(h):
    #each line is of the format <number of subjective sentences> followed by their indices
    hidden_string = ''
    for h_s in h:
        hidden_string += '{} '.format(len(h_s))
        hidden_string += ('{} '* len(h_s)).format(*h_s)
        hidden_string += '\n'

    return hidden_string

def read_hidden_variables(hidden_string):
    #each line is of the format <number of subjective sentences> followed by their indices
    h = []
    for line in hidden_string.splitlines():
        hidden_settings = line.split()
        num_hidden = int(hidden_settings[0])
        h_s = [int(i) for i in hidden_settings[1:]]
        h.append(h_s)
    return h

def sparse_to_row_dict(x):
    cx = scipy.sparse.coo_matrix(x)
    ret = collections.defaultdict(list)
    
    for i,j,v in zip(cx.row, cx.col, cx.data):
        ret[i].append([j,v])

    for i in ret:
        ret[i] = sorted(ret[i], key=lambda x:x[0])
        
    return ret

def write_features(X_ss, X_sp, X_dp, y=None):
    '''
    EXAMPLE
    -1 3
    0 1:1 3:0.5 5:1 S2:1 S4:1
    1 2:1 3:1 S1:0.8
    2 4:1 5:1 6:1 S2:1 S3:1
    3 1:0.5 2:0.1 3:1.5 4:0.2

    '''
    feature_string = ''

    for i in range(len(X_ss)):
        X_ss_dict = sparse_to_row_dict(X_ss[i])
        X_sp_dict = sparse_to_row_dict(X_sp[i])
        X_dp_dict = sparse_to_row_dict(X_dp[i])
        
        if y is not None:
            feature_string += '{} '.format(y[i])
        else:
            feature_string += '-1 '
        feature_string += '{}\n'.format(len(X_ss_dict))

        for s in range(len(X_ss_dict)):
            feature_string += '{} '.format(s)
            for k,v in X_sp_dict[s]:
                feature_string += '{}:{} '.format(k,v)
            for k,v in X_ss_dict[s]:
                feature_string += 'S{}:{} '.format(k,v)
            feature_string += '\n'

        feature_string += '{} '.format(len(X_ss_dict))
        for k,v in X_dp_dict[0]:
            feature_string += '{}:{} '.format(k,v)
        feature_string += '\n\n'
            
    return feature_string

def process_feature_line(line, subjective=False):
    features = line.split()
    ret = {}
    for feature in features[1:]:
        key,value = feature.split(':')
        if subjective and key[0] == 'S':
            ret[int(key[1:])] = float(value)
        elif not subjective and key[0] != 'S':
            ret[int(key)] = float(value)
            
    return ret

def dict_list_to_sparse(dicts):
    num_features = max(max(i.keys()) if len(i) else 0 for i in dicts) + 1
    ret = np.zeros((len(dicts), num_features))
    for i in range(len(dicts)):
        for k,v in dicts[i].items():
            ret[i,k] = v
    return ret

def read_features(features_string):
    data = features_string.splitlines()
    i = 0
    X_ss = []
    X_sp = []
    X_dp = []
    y = []
    while i < len(data):
        label, num_sentences = data[i].split()
        y.append(int(label))
        num_sentences = int(num_sentences)

        subjective_dicts = []
        polarity_dicts = []
        for j in range(num_sentences):
            s = data[i + j + 1]
            subjective_dicts.append(process_feature_line(s, subjective=True))
            polarity_dicts.append(process_feature_line(s))
        X_ss.append(dict_list_to_sparse(subjective_dicts))
        X_sp.append(dict_list_to_sparse(polarity_dicts))
        
        d = process_feature_line(data[i + num_sentences + 1])
        X_dp.append(dict_list_to_sparse([d]).flatten())
        
        i = i + num_sentences + 3
        
    return X_ss, X_sp, X_dp, y

def read_document_scores(predictions_string):
    ret = []
    for line in predictions_string.splitlines():
        if line.startswith('DOCUMENT POLARITY SCORE:'):
            ret.append(float(line.replace('DOCUMENT POLARITY SCORE:', '')))
    return ret
    
def read_predictions(predictions_string):
    ret = []
    for line in predictions_string.splitlines():
        labels = line.split()
        ret.append(int(labels[0]))
    return ret

def fit(feature_string, hidden_string, param_string, verbose=False):
    feature_file = tempfile.NamedTemporaryFile(dir=binDir)#, delete=False)
    feature_file.write(feature_string)
    feature_file.flush()
        
    hidden_file = tempfile.NamedTemporaryFile(dir=binDir)#, delete=False)
    hidden_file.write(hidden_string)
    hidden_file.flush()
    
    model_filename = feature_file.name + '_model'
    
    command = learnCommandTemplate.format(param_string,
                                          feature_file.name,
                                          hidden_file.name,
                                          model_filename)

    if verbose:
        p = subprocess.Popen(command.split())
    else:
        p = subprocess.Popen(command.split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    p.communicate()
    
    with open(model_filename) as f:
        model_string = f.read()
    os.unlink(model_filename)
        
    feature_file.close()
    hidden_file.close()
        
    return model_string

def predict(feature_string, model_string, latent=False, scores=False, verbose=False):
    feature_file = tempfile.NamedTemporaryFile(dir=binDir)
    feature_file.write(feature_string)
    feature_file.flush()
    
    model_file = tempfile.NamedTemporaryFile(dir=binDir)
    model_file.write(model_string)
    model_file.flush()
    
    output_filename = feature_file.name + '_output'

    param_string = ''
    if latent:
        param_string = '-l'
    elif scores:
        param_string = '-a'
        
    command = classifyCommandTemplate.format(param_string,
                                             feature_file.name,
                                             model_file.name,
                                             output_filename)

    if verbose:
        p = subprocess.Popen(command.split())
    else:
        p = subprocess.Popen(command.split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    p.communicate()

    with open(output_filename) as f:
        output_string = f.read()
    os.unlink(output_filename)
    
    if latent:
        return read_hidden_variables(output_string)
    elif scores:
        return read_document_scores(output_string)
    return read_predictions(output_string)

class LatentSVM:
    def __init__(self, num_iters=10, C=1.0,
                 extraction_size=30, feature_mode=3,
                 norm_mode=2, window_mode=False,
                 verbose=False,
                 feature_extractor=None):

        self.num_iters = num_iters
        self.extraction_size = extraction_size/100.
        self.param_string = write_params(C, extraction_size,
                                         feature_mode, norm_mode,
                                         window_mode)
        
        self.feature_extractor = feature_extractor #if we need to update the features between iters
        
        self.verbose = verbose
        
        self.model_string = None
        
    def fit(self, X_ss, X_sp, X_dp, y, h_init=None):
        #X_ss and X_sp is a list of S_i x F matrices
        #(not a tensor, as each document may have diff number of sentences)
        #X_dp is a D x F matrix, where D is the number of documents and F is the number of features

        self.model_string = None
        
        h = generate_hidden_variables(h_init, X_ss, self.extraction_size)
                
        for i in range(self.num_iters):
            if self.feature_extractor is not None:
                #update any features that depend on the hidden variables
                X_ss, X_sp, X_dp = self.feature_extractor.update(X_ss, X_sp, X_dp, h)

            #generate string for hidden vars
            h_string = write_hidden_variables(h)
                
            #generate string for training
            f_string = write_features(X_ss, X_sp, X_dp, y)
                
            #call svm_learn with the parameters
            self.model_string = fit(f_string, h_string, self.param_string, verbose=self.verbose)
            
            #call predict_hidden with the new model
            h = predict(f_string, self.model_string, latent=True, verbose=self.verbose)

        #read in the the final parameter file and return that
        return self.model_string
    
    def predict(self, X_ss, X_sp, X_dp, y=None):
        #call svm_classify
        f_string = write_features(X_ss, X_sp, X_dp, y)
        return predict(f_string, self.model_string, verbose=self.verbose)

    def predict_hidden(self, X_ss, X_sp, X_dp, y=None):
        #call svm_classify to predict the hidden variables
        f_string = write_features(X_ss, X_sp, X_dp, y)
        return predict(f_string, self.model_string, latent=True, verbose=self.verbose)

    def decision_function(self, X_ss, X_sp, X_dp):
        #call svm_classify with verbose and only return document scores
        f_string = write_features(X_ss, X_sp, X_dp)
        return predict(f_string, self.model_string, scores=True, verbose=self.verbose)
        
