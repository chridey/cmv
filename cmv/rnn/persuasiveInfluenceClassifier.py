import collections

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from cmv.rnn.persuasiveInfluenceRNN import PersuasiveInfluenceRNN

class PersuasiveInfluenceClassifier(BaseEstimator):
    def __init__(self, V, d, max_post_length, max_sentence_length,
                 max_title_length=256,
                 embeddings=None,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 num_layers=1,
                 learning_rate=0.01,
                 add_biases=False,
                 word_dropout=0,
                 dropout=0,
                 batch_size=100,
                 num_epochs=30,
                 verbose=False,
                 lambda_w=0,
                 early_stopping_heldout=0,
                 balance=False,
                 op=False,
                 hops=3,
                 outputfile='',
                 pairwise=False):

        print('outputfile', outputfile)
        
        self.V = V
        self.d = d
        self.max_post_length = max_post_length
        self.max_sentence_length = max_sentence_length
        self.max_title_length = max_title_length
        self.embeddings = embeddings
        self.GRAD_CLIP = GRAD_CLIP
        self.freeze_words = freeze_words
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.add_biases=False

        self.classifier = PersuasiveInfluenceRNN(V, d, max_post_length, max_sentence_length, max_title_length,
                                                 embeddings, GRAD_CLIP, freeze_words, num_layers, learning_rate,
                                                 add_biases, hops=hops, op=op)

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.lambda_w = lambda_w
        self.word_dropout = word_dropout
        self.dropout = dropout
        
        self.verbose = verbose
        self.early_stopping_heldout = early_stopping_heldout
        self.balance = balance
        self.outputfile = outputfile
        self.pairwise = pairwise
        
    def fit(self, X, y, X_heldout, y_heldout):
        if self.verbose:
            print(self.dropout, self.lambda_w, self.num_layers, self.word_dropout)
            print(collections.Counter(y))

        if self.early_stopping_heldout:
            X, X_heldout, y, y_heldout = train_test_split(X,
                                                          y,
                                                          test_size=self.early_stopping_heldout,
                                                          )
            print('Train Fold: {} Heldout: {}'.format(collections.Counter(y), collections.Counter(y_heldout)))

        #data = X zip(*X)
        #X = np.array(data[0])
        #num_batches = X.shape[0] // self.batch_size
        best = 0
        #training = np.array(zip(*data))
        num_batches = X[0].shape[0] // self.batch_size
        skf = StratifiedKFold(n_splits=num_batches+1, shuffle=True)
        folds = list(skf.split(X[0], y))
        
        for epoch in range(self.num_epochs):
            epoch_cost = 0

            #idxs = np.random.choice(X.shape[0], X.shape[0], False)
            #idxs = np.random.choice(X[0].shape[0], X[0].shape[0], False)
            #TODO: do stratified selection?            
            
            #if self.verbose:
            #    print('Unique', len(set(idxs)))
                
            for batch_num in range(num_batches+1):
                #s = self.batch_size * batch_num
                #e = self.batch_size * (batch_num+1)
                fold = folds[batch_num][1]
                
                inputs = []
                for input in X:
                    #tmp = [input[idxs[i]] for i in range(s,min(e,X[0].shape[0]))]
                    tmp = [input[fold[i]] for i in range(fold.shape[0])]
                    inputs.append(np.array(tmp))
                    #print(inputs[-1].shape)
                    
                #batch = training[idxs[s:e]]
                #inputs = zip(*batch)

                if self.verbose:
                    print('Y Batch:', collections.Counter(inputs[-1]))

                if self.word_dropout:
                    inputs[1] = inputs[1]*(np.random.rand(*(np.array(inputs[1]).shape)) < self.word_dropout)

                if not self.balance:
                    weights = np.ones(inputs[0].shape[0])
                else:
                    label_counts = collections.Counter(inputs[-1])
                    max_count = 1.*max(label_counts.values())
                    class_weights = {i:1/(label_counts[i]/max_count) for i in label_counts}
                    if self.verbose:
                        print(label_counts, class_weights)
                    weights = np.array([class_weights[i] for i in inputs[-1]]).astype(np.float32)

                #print([i.shape for i in inputs])
                cost = self.classifier.train(*(inputs+[self.lambda_w, self.dropout, weights]))
                if self.verbose:
                    print(epoch, batch_num, cost)
                epoch_cost += cost

            if self.early_stopping_heldout or (X_heldout is not None and y_heldout is not None):
                scores = self.decision_function(X_heldout[:-1]) #zip(zip(*X_heldout)[:-1]))
                #print(scores.shape, np.array(X_heldout[-1]).shape)
                score = roc_auc_score(X_heldout[-1], scores)  #zip(*X_heldout)[:-1], scores)
                if self.pairwise:
                    neg_scores = scores[scores.shape[0]//2:]
                    pos_scores = scores[:scores.shape[0]//2]
                    score = np.mean(pos_scores > neg_scores)
                if self.verbose:
                    print('{} ROC AUC: {}'.format(self.outputfile, score))
                    print('{} Accuracy: {}'.format(self.outputfile, accuracy_score(X_heldout[-1], scores > .5)))
                    print('{} Fscore: {}'.format(self.outputfile, precision_recall_fscore_support(X_heldout[-1], scores > .5)))
                    if self.pairwise:
                        print('{} Pairwise: {}'.format(self.outputfile, score))
                        
                if score > best:
                    best = score
                    best_params = self.classifier.get_params()

            if self.verbose:
                print(epoch_cost)

        if best > 0:
            self.classifier.set_params(best_params)

        return self
    
    def predict(self, X):
        scores = self.decision_function(X)
        return scores > .5
    
    def decision_function(self, X):
        scores = []

        num_batches = X[0].shape[0] // self.batch_size
        for batch_num in range(num_batches+1):        
            inputs = []
            for input in X:
                s = batch_num*self.batch_size
                e = (batch_num+1)*self.batch_size
                tmp = [input[i] for i in range(s,min(e,X[0].shape[0]))]
                inputs.append(np.array(tmp))
            scores.extend(list(self.classifier.predict(*inputs)))
        
        return np.array(scores)

    def save(self, outfilename):
        self.classifier.save(outfilename)
