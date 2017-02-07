import argparse
import itertools
import json

import nltk
import scipy
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from cmv.featureExtraction.featureExtractor import ArgumentFeatureExtractor
from cmv.preprocessing.thread import Thread
from cmv.preprocessing.dataIterator import DataIterator, PairedDataIterator

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
    parser.add_argument('--verbose', action='store_true')    
    
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
            featureLabels[subset] = featureLabels[subset].append(features, ignore_index=True)
            
    if args.features_file:
        with open(args.features_file, 'w') as f:
            json.dump({'{}_features'.format(i): featureLabels[i].fillna(0).to_json() for i in ('train', 'val')}, f)

    keys = list(set(featureLabels['train'].keys()) - {'label'})
    X = featureLabels['train'][keys]
    X_val = featureLabels['train'][keys]
    labels = featureLabels['train']['label']
    labels_val = featureLabels['train']['label']

    Cs = [.0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    norms = ['l1', 'l2']
    best = 0
    best_params = {'C': None, 'penalty': None}
        
    for C in Cs:
        for norm in norms:
            lr = LogisticRegression(C=C, penalty=norm, class_weight='balanced')
            lr.fit(X, labels)
            predictions = lr.predict(X_val)
            
            if args.accuracy:
                score = accuracy_score(labels, predictions)
            elif args.pairwise:
                scores = lr.predict_proba(X_val)[:,0].flatten()
                neg_scores = scores[1::2]
                pos_scores = scores[::2]
                score = np.mean(pos_scores > neg_scores)
            elif args.auc:
                lr_scores = lr.decision_function(X_val)
                score = roc_auc_score(labels_val, lr_scores)
            else:
                score = accuracy_score(labels, predictions)
                
            if score > best:
                best = score
                best_params = dict(C=C, penalty=norm)

            if args.verbose:
                print(C, norm, score)
                
    lr = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], class_weight='balanced')
    lr.fit(X, labels)
    
    if args.biases_file:
        with open(args.biases_file, 'w') as f:
            json.dump([lr.decision_function(X).tolist(), lr.decision_function(X_val).tolist()], f)

    if args.predictions_file:        
        np.save(args.predictions_file, lr.predict(X_val))            
