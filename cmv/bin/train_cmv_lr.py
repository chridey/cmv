import argparse
import itertools
import json

import nltk
import scipy
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.externals import joblib

from cmv.featureExtraction.featureExtractor import ArgumentFeatureExtractor
from cmv.preprocessing.thread import Thread
from cmv.preprocessing.dataIterator import DataIterator, PairedDataIterator

def score_function(X, labels, predictions, accuracy=False, pairwise=False, auc=False):
    if accuracy:
        score = accuracy_score(labels, predictions)
    elif auc:
        lr_scores = lr.decision_function(X)
        score = roc_auc_score(labels, lr_scores)        
    elif pairwise:
        scores = lr.predict_proba(X)[:,0].flatten()
        pos_scores = scores[::2]
        neg_scores = scores[1::2]
        score = np.mean(pos_scores > neg_scores)
    else:
        score = accuracy_score(labels, predictions)
        
    return score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an influence classifier')
    parser.add_argument('metadata_file')
    parser.add_argument('--features_file')
    parser.add_argument('--predictions_file')
    parser.add_argument('--biases_file')
    parser.add_argument('--model_file')    
    parser.add_argument('--cv', action='store_true')
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

    featureLabels = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
    for subset in ('train', 'val', 'test'):
        if subset not in data:
            print('WARNING: {} not in data'.format(subset))
            continue
        di = diType(data[subset])
        for thread,label in di.iterPosts():
            features = fe.addFeatures(thread)
            features['label'] = label
            featureLabels[subset] = featureLabels[subset].append(features, ignore_index=True)
            
    if args.features_file:
        with open(args.features_file, 'w') as f:
            json.dump({'{}_features'.format(i): featureLabels[i].fillna(0).to_json() for i in ('train', 'val')}, f)

    keys = list(set(featureLabels['train'].keys()) - {'label'})
    X = featureLabels['train'][keys]
    print(X.iloc[0])
    X_val = featureLabels['val'][keys]
    labels = featureLabels['train']['label']
    labels_val = featureLabels['val']['label']
    
    Cs = [.0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
    norms = ['l1', 'l2']
    best = 0
    best_params = {'C': None, 'penalty': None}
        
    for C in Cs:
        for norm in norms:
            lr = LogisticRegression(C=C, penalty=norm, class_weight='balanced')
            lr.fit(X, labels)
            predictions = lr.predict(X_val)

            score = score_function(X_val, labels_val, predictions, args.accuracy, args.paired, args.auc)
                
            if score > best:
                best = score
                best_params = dict(C=C, penalty=norm)

            if args.verbose:
                print(C, norm, score)
                
    lr = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], class_weight='balanced')
    lr.fit(X, labels)

    if 'test' in data:
        predictions = lr.predict(featureLabels['test'][keys])
        score = score_function(featureLabels['test'][keys], featureLabels['test']['label'],
                               predictions, args.accuracy, args.paired, args.auc)                     
        print('Test Score: {}'.format(score))

    if args.model_file:
        joblib.dump(lr, args.model_file)
            
    if args.biases_file:
        with open(args.biases_file, 'w') as f:
            output = dict(train=lr.decision_function(X).tolist(),
                          val=lr.decision_function(X_val).tolist())
            if 'test' in data:
                output.update(test=lr.decision_function(featureLabels['test'][keys]).tolist())
            json.dump(output, f)

    if args.predictions_file:        
        np.save(args.predictions_file, lr.predict(X_val))            
