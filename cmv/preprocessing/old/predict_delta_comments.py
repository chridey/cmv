import subprocess
import os
import sys
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support


from spacy.en import English

cmv_pattern = re.compile('cmv:?', re.IGNORECASE)
cmv_pattern2 = re.compile('chang[e|i][s|d]?(ng)? my view', re.IGNORECASE)
delta_pattern = re.compile('delta', re.IGNORECASE)
delta_pattern2 = re.compile('8710', re.IGNORECASE)
delta_pattern3 = re.compile(u'\u2206', re.IGNORECASE)

nlp = English()

def cleanup(text):

    parsed_text = [nlp(unicode(i)) for i in text.split('\n')]
    ret = []
    for par in parsed_text:
        for sent in par:
            if cmv_pattern.match(sent.string) or cmv_pattern2.match(sent.string) or delta_pattern.match(sent.string) or delta_pattern2.match(sent.string) or delta_pattern3.match(sent.string):
                continue
            ret.append(sent.string)
    return '\n'.join(ret)

class CMVReader:
    def __init__(self, indir):
        self.indir = indir
        self._labels = None
        
    def iterFiles(self):
        for label in os.listdir(self.indir):
            full_path_dir = os.path.join(self.indir, label)
            for filename in os.listdir(full_path_dir):
                full_path_filename = os.path.join(full_path_dir, filename)
                yield label,full_path_filename

    def iterData(self):
        for label,filename in self.iterFiles():
            with open(filename) as f:
                print(filename)
                for line in f:
                    j = json.loads(line)
                    if 'body' in j:
                        yield label,j

    def __iter__(self):
        for label,data in self.iterData():
            yield cleanup(data['body'])
            
    @property
    def labels(self):
        if self._labels is None:
            self._labels = []
            for label,data in self.iterData():
                print(label)
                self._labels.append(label == 'positives')
                    
        return self._labels

if __name__ == '__main__':
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]

    train = CMVReader(train_dir)
    test = CMVReader(test_dir)

    train.labels
    test.labels
    
    vect_bow = TfidfVectorizer(use_idf=True, norm='l2', min_df=5, ngram_range=(1,3))
    print('vectorizing train...')
    train_bow = vect_bow.fit_transform(train)
    print('vectorizing test...')
    test_bow = vect_bow.transform(test)

    print(train_bow.shape, test_bow.shape)
    print('training...')
    for C in [.0000001, .000001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]:
        for pen in ['l1', 'l2']:
            lr = LogisticRegression(C=C, penalty=pen, class_weight = 'balanced')
            lr.fit(train_bow, train.labels)
            lr_predictions = lr.predict(test_bow)
            acc = accuracy_score(test.labels, lr_predictions)
            lr_scores = lr.decision_function(test_bow)
            auc_score = roc_auc_score(test.labels, lr_scores)
            precision, recall, fscore, _ = precision_recall_fscore_support(test.labels, lr_predictions)
            print(C, pen, acc, auc_score, precision, recall, fscore)
    
