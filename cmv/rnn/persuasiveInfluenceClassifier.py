import collections

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from cmv.rnn.persuasiveInfluenceRNN import PersuasiveInfluenceRNN

class PersuasiveInfluenceClassifer(BaseEstimator):
    def __init__(self, V, d, max_post_length, max_sentence_length, embeddings=None,
                 rd=100,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 num_layers=1,
                 learning_rate=0.01,
                 V_frames=0,
                 d_frames=0,
                 V_intra=0,
                 d_intra=0,
                 d_inter=0,
                 V_sentiment=0,
                 d_sentiment=0,
                 add_biases=False,
                 highway=True,
                 word_dropout=0,
                 batch_size=100,
                 num_epochs=30,
                 verbose=False,
                 lambda_w=0,
                 early_stopping_heldout=0,
                 balance=False):
        self.V = V
        self.d = d
        self.rd = rd
        self.max_post_length = max_post_length
        self.max_sentence_length = max_sentence_length
        self.embeddings = embeddings
        self.GRAD_CLIP = GRAD_CLIP
        self.freeze_words = freeze_words
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.V_frames = V_frames
        self.d_frames = d_frames
        self.V_intra = V_intra
        self.d_intra = d_intra
        self.d_inter = d_inter
        self.V_sentiment = V_sentiment
        self.d_sentiment = d_sentiment
        self.add_biases=False
        self.highway = highway

        self.classifier = PersuasiveInfluenceRNN(V, d, rd, max_post_length, max_sentence_length, embeddings,
                                                 GRAD_CLIP, freeze_words, num_layers, learning_rate, V_frames,
                                                 d_frames, V_intra, d_intra, d_inter, V_sentiment, d_sentiment,
                                                 add_biases, highway)

        self.lambda_w = lambda_w
        self.word_dropout = word_dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.early_stopping_heldout = early_stopping_heldout
        self.balance = balance
        
    def fit(self, X, y):
        if self.verbose:
            print(self.dropout, self.lambda_w, self.class_weights,
                  self.freeze_words, self.num_hidden, self.word_dropout)
            print(collections.Counter(y))

        if self.early_stopping_heldout:
            X, X_heldout, y, y_heldout = train_test_split(X,
                                                          y,
                                                          test_size=self.early_stopping_heldout,
                                                          )
            print('Train Fold: {} Heldout: {}'.format(collections.Counter(y), collections.Counter(y_heldout)))

        data = zip(*X)
        X = data[0]
        num_batches = X.shape[0] // self.batch_size
        best = 0
        training = np.array(zip(*data))
        
        for epoch in range(self.num_epochs):
            epoch_cost = 0

            idxs = np.random.choice(X.shape[0], X.shape[0], False)
            if self.verbose:
                print('Unique', len(set(idxs)))
                
            for batch_num in range(num_batches+1):
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)

                batch = training[idxs[s:e]]
                inputs = zip(*batch)

                print('Y Batch:', collections.Counter(y_batch))

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

                cost = self.classifier.train(*(inputs+[self.lambda_w, self.dropout, weights]))
                if self.verbose:
                    print(epoch, batch_num, cost)
                epoch_cost += cost

            if self.early_stopping_heldout:
                scores = self.decision_function(zip(zip(*X_heldout)[:-1]))
                auc_score = roc_auc_score(zip(*X_heldout)[:-1], scores)
                if self.verbose:
                    print('{} ROC AUC: {}'.format(outputfile, auc_score))
                    if score > best:
                        best = score
                        best_params = self.classifier.get_params()

            if self.verbose:
                print(r, epoch, cost)

        if best > 0:
            self.classifier.set_params(best_params)

        return self
    
    def predict(self, X):
        scores = self.decision_function(X)
        return scores > .5
    
    def decision_function(self, X):
        inputs = zip(*X)
        scores = self.classifier.predict(*inputs)
        return scores

    def save(self, outfilename):
        self.classifier.save(outfilename)
