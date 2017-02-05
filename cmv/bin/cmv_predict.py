import argparse
import itertools
import json

import nltk
import scipy
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression

from cmv.featureExtraction.featureExtractor import ArgumentFeatureExtractor
from cmv.preprocessing.thread import Thread
from cmv.preprocessing.dataIterator import DataIterator, PairedDataIterator

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

    parser = argparse.ArgumentParser(description='train an influence classifier')
    parser.add_argument('metadata_file')
    parser.add_argument('--features_file')
    parser.add_argument('--predictions_file')
    parser.add_argument('--biases_file')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--auc', action='store_true')
    parser.add_argument('--paired', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.metadata_file) as f:
        data = json.load(f)

    if args.paired:
        diType = PairedDataIterator
    else:
        diType = DataIterator
    
    fe = ArgumentFeatureExtractor()

    featureLabels = {'train': pd.DataFrame(), 'val': pd.DataFrame()}
    for subset in ('train', 'val'):
        di = diType(data, subsets = [subset])
        for thread,label in di.iterPosts():
            features = fe.addFeatures(thread)
            features['label'] = label
            featureLabels[subset].append(features, ignore_index=True)

    if args.features_file:
        with open(args.features_file) as f:
            json.dump({'{}_features'.format(i): featureLabels[i].fillna(0).to_json() for i in ('train', 'val')}, f)

    keys = list(set(df.keys()) - {'label'})
    X = featureLabels['train'][keys]
    X_val = featureLabels['train'][keys]
    labels = featureLabels['train']['label']
    labels_val = featureLabels['train']['label']

    Cs = [.0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    norms = ['l1', 'l2']
    best = 0
    best_params = {'C': None, 'pen': None}
        
    for C in Cs:
        for norm in norms:
            lr = LogisticRegression(C=C, penalty=pen, class_weight='balanced')
            lr.fit(X, labels)
            predictions = lr.predict(X_val, labels_val)
            
            if args.accuracy:
                score = accuracy_score(labels, predictions)
            elif args.pairwise:
                scores = lr.predict_proba(X_val)[:,0].flatten()
                neg_scores = scores[1::2]
                pos_scores = scores[::2]
                score = np.mean(pos_scores > neg_scores)
            elif args.auc:
                score = roc_auc_score(lr_labels, lr_scores)
            if score > best:
                best = score
                best_params = dict(C=C, penalty=norm)

    lr = LogisticRegression(C=C, penalty=pen, class_weight='balanced')
    lr.fit(X, labels)
    
    if args.biases_file:
        with open(args.biases_file) as f:
            json.dump([lr.decision_function(X).tolist(), lr.decision_function(X_val)], f)

    if args.predictions_file:        
        np.save(args.predictions_file, lr.predict(X_val))            
