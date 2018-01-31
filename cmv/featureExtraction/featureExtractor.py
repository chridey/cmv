import pandas as pd
import nltk

def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

class ArgumentFeatureExtractor:
    '''features for an entire document'''
    def __init__(self,
                 settings=None,
                 verbose=False):

        if settings is not None:
            self.settings = settings
        else:
            self.settings = {'featureSettings': {'interplay': True}}
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
                
        self.validFeatures = {'interplay': self.getInterplay,
                              }
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

    def getInterplay(self, dataPoint):
        op_all = set(dataPoint.originalPost.getAllWords(True))
        rr_all = set(dataPoint.response.getAllWords(True))
        op_stop = op_all & self.stopwords
        rr_stop = rr_all & self.stopwords
        op_content = op_all - self.stopwords
        rr_content = rr_all - self.stopwords

        key = self.functionFeatures[self.getInterplay]
        all_interplay = calculate_interplay(op_all, rr_all)
        stop_interplay = calculate_interplay(op_stop, rr_stop)
        content_interplay = calculate_interplay(op_content, rr_content)
        keys = [key + '_int', key + '_reply_frac', key + '_op_frac', key + '_jaccard']
        keys = ['all_' + i for i in keys] + ['stop_' + i for i in keys] + ['content_' + i for i in keys]
        return zip(keys, all_interplay + stop_interplay + content_interplay)
        
    def addFeatures(self, dataPoint, featureSettings=None):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']
            
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint))
        return features
        
        return features

