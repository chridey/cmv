import collections
import json

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from cmv.rnn.persuasiveInfluenceRNN import PersuasiveInfluenceRNN, load as rnn_load

class PersuasiveInfluenceClassifier(BaseEstimator):
    def __init__(self,
                 vocab,
                 rnn_params,
                 batch_size=100,
                 num_epochs=30,                 
                 lambda_w=0,                 
                 word_dropout=0,
                 dropout=0,
                 early_stopping_heldout=0,
                 balance=False,
                 pairwise=False,
                 verbose=False):

        self.vocab = vocab
        
        self.model = PersuasiveInfluenceRNN(**rnn_params)

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.lambda_w = lambda_w
        self.word_dropout = word_dropout
        self.dropout = dropout
        
        self.early_stopping_heldout = early_stopping_heldout
        self.balance = balance
        self.pairwise = pairwise
        self.verbose = verbose
                
    def fit(self, X, y, X_heldout, y_heldout):
        if self.verbose:
            print(self.dropout, self.lambda_w, self.word_dropout)
            print(collections.Counter(y))

        if self.early_stopping_heldout:
            X, X_heldout, y, y_heldout = train_test_split(X,
                                                          y,
                                                          test_size=self.early_stopping_heldout,
                                                          )
            print('Train Fold: {} Heldout: {}'.format(collections.Counter(y), collections.Counter(y_heldout)))

        best = 0
        num_batches = X[0].shape[0] // self.batch_size
        skf = StratifiedKFold(n_splits=num_batches+1, shuffle=True)
        folds = list(skf.split(X[0], y))
        
        for epoch in range(self.num_epochs):
            epoch_cost = 0

            for batch_num in range(num_batches+1):
                fold = folds[batch_num][1]
                
                inputs = []
                for input in X:
                    tmp = [input[fold[i]] for i in range(fold.shape[0])]
                    inputs.append(np.array(tmp))

                inputs.append(np.array(y)[fold])

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

                cost = self.model.train(*(inputs+[self.lambda_w, self.dropout, weights]))
                if self.verbose:
                    print(epoch, batch_num, cost)
                epoch_cost += cost

            if self.early_stopping_heldout or (X_heldout is not None and y_heldout is not None):
                score = self.get_score(X_heldout, y_heldout)
                
                if score > best:
                    best = score
                    best_params = self.model.get_params()

            if self.verbose:
                print(epoch_cost)

        if best > 0:
            self.model.set_params(best_params)

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
            scores.extend(list(self.model.predict(*inputs)))
        
        return np.array(scores)

    def get_score(self, X, y):
        scores = self.decision_function(X)
        score = roc_auc_score(y, scores)
        
        if self.verbose:
            print('{} ROC AUC: {}'.format(self.pretty_params, score))
            print('{} Accuracy: {}'.format(self.pretty_params, accuracy_score(y, scores > .5)))
            print('{} Fscore: {}'.format(self.pretty_params, precision_recall_fscore_support(y, scores > .5)))

        if self.pairwise:
            neg_scores = scores[scores.shape[0]//2:]
            pos_scores = scores[:scores.shape[0]//2]
            score = np.mean(pos_scores > neg_scores)
            
            if self.verbose:
                print('{} Pairwise: {}'.format(self.pretty_params, score))
                
        return score

    @property
    def params(self):
        return dict(lambda_w=self.lambda_w,
                    word_dropout=self.word_dropout,
                    dropout=self.word_dropout,
                    rnn_params=self.model.hyper_params)

    @property
    def pretty_params(self):
        ret = []
        for key,value in self.params.items():
            if key == 'rnn_params':
                for key,value in self.params[key].items():
                    ret.append('{}={}'.format(key, value))
            else:
                ret.append('{}={}'.format(key, value))
        return '_'.join(ret)
    
    def save(self, outfilename):
        
        with open(outfilename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

        with open(outfilename + '.params', 'w') as f:
            json.dump(self.params, f)
            
        self.model.save(outfilename +'.model')
        
def load(filename, verbose=False):
    
    with open(filename + '.vocab') as f:
        vocab = json.load(f)
    
    with open(filename + '.params') as f:
        params = json.load(f)

    params['vocab'] = vocab
    params['verbose'] = verbose
    classifier = PersuasiveInfluenceClassifier(**params)

    rnn_load(classifier.model, filename + '.model.npz')

    return classifier
