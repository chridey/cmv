import argparse
import itertools

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs
from cmv.preprocessing.preprocess import normalize_from_body

from cmv.ml.latentSVM import LatentSVM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

import nltk
import scipy
import numpy as np

from spacy.en import English
nlp = English()

def handle_input(pairs):
    op_text = []
    neg_text = []
    pos_text = []

    for pair in pairs:
        post = normalize_from_body(pair['op_text'], True)
        op_text.append(post)
    
        post = normalize_from_body(pair['negative']['comments'][0]['body'])
        neg_text.append(post)
        
        post = normalize_from_body(pair['positive']['comments'][0]['body'])
        pos_text.append(post)

    return op_text,neg_text,pos_text

def handle_preprocessed():
    import json
    
    filename = '/local/nlp/chidey/cmv/pair_task/train_pair_data.jsonlist.bz2.out'
    with open(filename) as f:
        preprocessed = json.load(f)

    op_tok = []
    rr_tok = []
    labels = []
    h_scores = []
    for datum in preprocessed:
        if len(datum['positive']):
            rr_tok.append(datum['positive'][0][0]['sentences'])
            scores = datum['positive'][0][2]
            if not len(scores):
                scores = [1. for i in datum['positive'][0][0]['sentences']]
            assert(len(scores) == len(datum['positive'][0][0]['sentences']))
            h_scores.append(scores)
            op_tok.append(datum['op']['sentences'])
            labels.append(1)
        if len(datum['negative']):
            rr_tok.append(datum['negative'][0][0]['sentences'])
            scores = datum['negative'][0][2]
            if not len(scores):
                scores = [1. for i in datum['negative'][0][0]['sentences']]
            assert(len(scores) == len(datum['negative'][0][0]['sentences']))               
            h_scores.append(scores)
            op_tok.append(datum['op']['sentences'])
            labels.append(-1)            
    return op_tok, rr_tok, labels, h_scores

def tokenize_sentences(op, rr):
    op_tok = []
    rr_tok = []
    for i in range(len(op)):
        #op_sent_tok = [nltk.word_tokenize(j) for j in nltk.sent_tokenize(op[i])]
        #rr_sent_tok = [nltk.word_tokenize(j) for j in nltk.sent_tokenize(rr[i])]
        op_sent_tok = [[j.string.strip() for j in sent] for sent in nlp(unicode(op[i])).sents]
        rr_sent_tok = [[j.string.strip() for j in sent] for sent in nlp(unicode(rr[i])).sents]

        op_tok.append(op_sent_tok)
        rr_tok.append(rr_sent_tok)
    return op_tok, rr_tok

stopwords = set(nltk.corpus.stopwords.words('english'))
def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

def make_interplay_features(op, rr):
    op_all = set(op)
    rr_all = set(rr)
    op_stop = op_all & stopwords
    rr_stop = rr_all & stopwords
    op_content = op_all - stopwords
    rr_content = rr_all - stopwords
    
    return calculate_interplay(op_all, rr_all) + calculate_interplay(op_stop, rr_stop) + calculate_interplay(op_content, rr_content)
                                                
def transform_interplay(op, rr):
    doc = []
    for i in range(len(op)):
        doc.append(make_interplay_features(itertools.chain(*op[i]),
                                           itertools.chain(*rr[i])))
    return doc

def transform_sentences(op, rr, transformer, interplay=False, h_scores=None):
    X_ss = []
    X_sp = []
    h = []
    for i in range(len(op)):
        features = []
        h_s = []
        for j in range(len(rr[i])):
            try:
                transformed_features = transformer.transform([rr[i][j]])
            except ValueError:
                continue
            if transformed_features.nnz == 0:
                continue
            if interplay:
                interplay_features = make_interplay_features(itertools.chain(*op[i]),
                                                             rr[i][j])
                transformed_features = combine_features([transformed_features,
                                                         interplay_features])
            features.append(transformed_features)
            if h_scores is not None:
                h_s.append(h_scores[i][j])
        features = scipy.sparse.vstack(features)
        X_ss.append(features)
        X_sp.append(features)

        if h_scores is not None:
            #order the results by their scores
            h.append(np.argsort(h_s)[::-1])

    return X_ss,X_sp,h

def combine_features(features):
    return scipy.sparse.csr_matrix(scipy.sparse.hstack([scipy.sparse.csr_matrix(i) for i in features]))

class MaintainOriginalPostKFold:
    def __init__(self, labels, n_folds=2, shuffle=False, random_state=None):
        self.labels = labels
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def __iter__(self):
        num_points = len(self.labels) // 2
        indices = np.random.choice(num_points, size=num_points, replace=False).tolist()
        batch_size = num_points // self.n_folds + 1
        for i in range(self.n_folds):
            test = indices[i*batch_size:(i+1)*batch_size]
            train = indices[:i*batch_size] + indices[(i+1)*batch_size:]
            yield train + [i+num_points for i in train], test + [i+num_points for i in test]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    args = parser.parse_args()

    if 0:
        train_pairs = load_train_pairs()    
        train_op, train_neg, train_pos = handle_input(train_pairs)
        train_op_tok, train_rr_tok = tokenize_sentences(train_op+train_op, train_neg+train_pos)
        labels_train = [-1]*len(train_neg)+[1]*len(train_pos)
    else:
        train_op_tok,train_rr_tok,labels_train,h_scores = handle_preprocessed()
        
    heldout_pairs = load_test_pairs()    
    heldout_op, heldout_neg, heldout_pos = handle_input(heldout_pairs)
    heldout_op_tok, heldout_rr_tok = tokenize_sentences(heldout_op+heldout_op, heldout_neg+heldout_pos)    

    vect = TfidfVectorizer(use_idf=False, norm='l2', min_df=5,
                           tokenizer=lambda x:[i.lower() for i in x if len(i) > 1],
                           lowercase=False)
    X_dp = vect.fit_transform([itertools.chain(*i) for i in train_rr_tok])
    int_dp = transform_interplay(train_op_tok,
                                 train_rr_tok)
    X_dp = combine_features([X_dp, int_dp])
    print(X_dp.shape)    
    X_ss, X_sp,h = transform_sentences(train_op_tok,
                                       train_rr_tok,
                                       vect,
                                       interplay=True,
                                       h_scores=h_scores)

    X_dp_test = vect.transform([itertools.chain(*i) for i in heldout_rr_tok])
    int_dp_test = transform_interplay(heldout_op_tok,
                                      heldout_rr_tok)
    X_dp_test = combine_features([X_dp_test, int_dp_test])
    print(X_dp_test.shape)
    X_ss_test,X_sp_test,h_test = transform_sentences(heldout_op_tok,
                                                    heldout_rr_tok,
                                                    vect,
                                                    interplay=True)
    labels_test = [-1]*len(heldout_neg)+[1]*len(heldout_pos)

    extraction_sizes = [10, 30, 50, 70, 90]
    Cs = [.0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    if len(h):
        inits = [h]
    else:
        inits = [None, 'random', 'first', 'last'] #
    for init in inits:
      for extraction_size in extraction_sizes:
          for C in Cs:
            if init is None or type(init) == str:
                print('Latent Init: {} Extraction Size: {} C: {}'.format(init, extraction_size, C))
            else:
                print('Extraction Size: {} C: {}'.format(extraction_size, C))
            model = LatentSVM(C=C, extraction_size=extraction_size)
            model.fit(X_ss, X_sp, X_dp, labels_train, h_init=init)

            predictions = model.predict(X_ss_test, X_sp_test, X_dp_test, labels_test)
            scores = np.array(model.decision_function(X_ss_test, X_sp_test, X_dp_test))
            neg_scores = scores[:len(scores)//2]
            pos_scores = scores[len(scores)//2:]
            ret = pos_scores > neg_scores
            print('Accuracy: {}'.format(accuracy_score(labels_test, predictions)))
            print('Pairwise Accuracy: {}'.format(1.*sum(ret)/len(ret)))
    
