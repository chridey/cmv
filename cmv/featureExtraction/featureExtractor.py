import math

import pandas as pd
import numpy as np
import nltk

from altlex.featureExtraction.baseFeatureExtractor import BaseFeatureExtractor
from altlex.utils import wordUtils

from cmv.featureExtraction.altlexHandler import AltlexHandler, countValidConnectives
from cmv.featureExtraction import config

def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

class ArgumentFeatureExtractor(BaseFeatureExtractor):
    '''features for an entire document'''
    def __init__(self,
                 settings=config.defaultConfig,
                 verbose=False,
                 cache=True):
        self.config = config.Config(settings)
        
        self.altlexHandler = AltlexHandler(**self.config.altlexSettings)

        self.stopwords = set(nltk.corpus.stopwords.words('english'))
                
        self.validFeatures = {'causal_pct': self.getCausalPct,
                              'noncausal_pct': self.getNonCausalPct,
                              'causal_score': self.getCausalScore,
                              'causal_altlex_pct': self.getCausalAltlexPct,
                              'noncausal_altlex_pct': self.getNonCausalAltlexPct,
                              'wordnet_response': self.getWordNetResponse,
                              'verbnet_response': self.getVerbNetResponse,
                              #title, all roots and arguments
                              'wordnet_title_response_interaction': self.getWordNetTitleResponse,
                              #OP, all roots and arguments
                              'wordnet_post_response_interaction': self.getWordNetPostResponse,
                              #title, all roots and arguments
                              'verbnet_title_response_interaction': self.getVerbNetTitleResponse,
                              #OP, all roots and arguments
                              'verbnet_post_response_interaction': self.getVerbNetPostResponse,
                              'framenet_response': self.getFrameNetResponse,
                              'framenet_altlex_sum': self.getFrameNetAltlexSum,
                              'interplay': self.getInterplay,
                              }
                              #'connective_patterns'
                              #intersections
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

        self.cache = {} if cache else None
        
    def getCausalPct(self, dataPoint):
        length = len(dataPoint.response.metadata)
        if not length:
            return {self.functionFeatures[self.getCausalPct]: 0}
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                 wordUtils.causal_markers)
        return {self.functionFeatures[self.getCausalPct]: countValidConnectives(sentences)/length}

    def getNonCausalPct(self, dataPoint):
        length = len(dataPoint.response.metadata)
        if not length:
            return {self.functionFeatures[self.getNonCausalPct]: 0}
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                 wordUtils.noncausal_markers)
        return {self.functionFeatures[self.getNonCausalPct]: countValidConnectives(sentences)/length}
    
    def getCausalAltlexPct(self, dataPoint):
        length = len(dataPoint.response.metadata)
        if not length:
            return {self.functionFeatures[self.getCausalAltlexPct]: 0}
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                 self.altlexHandler.causalAltlexes,
                                                 checkCache=True)
        
        return {self.functionFeatures[self.getCausalAltlexPct]: countValidConnectives(sentences)/length}
    
    def getNonCausalAltlexPct(self, dataPoint):
        length = len(dataPoint.response.metadata)
        if not length:
            return {self.functionFeatures[self.getNonCausalAltlexPct]: 0}
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                  self.altlexHandler.nonCausalAltlexes)
        return {self.functionFeatures[self.getNonCausalAltlexPct]: countValidConnectives(sentences)/length}

    def getCausalScore(self, dataPoint):
        length = len(dataPoint.response.metadata)
        if not length:
            return {self.functionFeatures[self.getCausalScore]: 0}
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                 self.altlexHandler.causalAltlexes,
                                                 checkCache=True)
        causal, noncausal = self.altlexHandler.causalScore(sentences, empty=False)
        return {self.functionFeatures[self.getCausalScore] + '_causal' : causal/length,
                self.functionFeatures[self.getCausalScore] + '_noncausal' : noncausal/length}
    
    #still use the lexical semantic features for interaction between title/OP and response
    #but sum over cartesian product and then L2 normalize

    def getWordNetResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        features = self.altlexHandler.getWordNetFullSentence(sentences)
        return {self.functionFeatures[self.getWordNetResponse] + '_' + k:v for (k,v) in features.items()}
        
    def getVerbNetResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        features = self.altlexHandler.getVerbNetFullSentence(sentences)
        return {self.functionFeatures[self.getWordNetResponse] + '_' + k:v for (k,v) in features.items()}
    
    def getWordNetTitleResponse(self, dataPoint):
        
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        title = self.altlexHandler.getConnectiveSentences(dataPoint.title.metadata,
                                                    self.altlexHandler.causalAltlexes)
        
        response_features = self.altlexHandler.getWordNetFullSentence(sentences, normalize=False)
        title_features = self.altlexHandler.getWordNetFullSentence(title, normalize=False)
        ret = {}
        for rk,rv in response_features.items():
            for tk,tv in title_features.items():
                ret[rk + '_' + tk] = rv*tv
        l2 = math.sqrt(sum(i**2 for i in ret.values()))

        return {self.functionFeatures[self.getWordNetTitleResponse] +'_' + k:v/l2 for (k,v) in ret.items()}
    
    def getVerbNetTitleResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        title = self.altlexHandler.getConnectiveSentences(dataPoint.title.metadata,
                                                    self.altlexHandler.causalAltlexes)
        
        response_features = self.altlexHandler.getVerbNetFullSentence(sentences, normalize=False)
        title_features = self.altlexHandler.getVerbNetFullSentence(title, normalize=False)
        ret = {}
        for rk,rv in response_features.items():
            for tk,tv in title_features.items():
                ret[rk + '_' + tk] = rv*tv
        l2 = math.sqrt(sum(i**2 for i in ret.values()))

        return {self.functionFeatures[self.getVerbNetTitleResponse] +'_' + k:v/l2 for (k,v) in ret.items()}
    
    def getWordNetPostResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        op = self.altlexHandler.getConnectiveSentences(dataPoint.originalPost.metadata,
                                                        self.altlexHandler.causalAltlexes)
        
        response_features = self.altlexHandler.getWordNetFullSentence(sentences, normalize=False)
        op_features = self.altlexHandler.getWordNetFullSentence(op, normalize=False)
        ret = {}
        for rk,rv in response_features.items():
            for tk,tv in op_features.items():
                ret[rk + '_' + tk] = rv*tv
        l2 = math.sqrt(sum(i**2 for i in ret.values()))

        return {self.functionFeatures[self.getWordNetPostResponse] +'_' + k:v/l2 for (k,v) in ret.items()}
    
    def getVerbNetPostResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        op = self.altlexHandler.getConnectiveSentences(dataPoint.originalPost.metadata,
                                                        self.altlexHandler.causalAltlexes)
        
        response_features = self.altlexHandler.getVerbNetFullSentence(sentences, normalize=False)
        op_features = self.altlexHandler.getVerbNetFullSentence(op, normalize=False)
        ret = {}
        for rk,rv in response_features.items():
            for tk,tv in op_features.items():
                ret[rk + '_' + tk] = rv*tv
        l2 = math.sqrt(sum(i**2 for i in ret.values()))

        return {self.functionFeatures[self.getVerbNetPostResponse] +'_' + k:v/l2 for (k,v) in ret.items()}
    
    def getFrameNetResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                            self.altlexHandler.causalAltlexes,
                                                            checkCache=True)
        causal_sum,anticausal_sum = self.altlexHandler.getFrameNetResponseSum(sentences)
        return {self.functionFeatures[self.getFrameNetResponse] + '_causal': causal_sum,
                self.functionFeatures[self.getFrameNetResponse] + '_anticausal': anticausal_sum}

    
    def getFrameNetAltlexSum(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response.metadata,
                                                        self.altlexHandler.causalAltlexes,
                                                        checkCache=True)
        causal_sum,anticausal_sum = self.altlexHandler.getFrameNetAltlexSum(sentences)
        return {self.functionFeatures[self.getFrameNetAltlexSum] + '_causal': causal_sum,
                self.functionFeatures[self.getFrameNetAltlexSum] + '_anticausal': anticausal_sum}

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
        features = super(ArgumentFeatureExtractor, self).addFeatures(dataPoint,
                                                                     featureSettings)

        if self.cache is not None:
            self.cache = {}

        self.altlexHandler.clearCache()
        
        return features

if __name__ == '__main__':
    import json
    import sys
    from cmv.featureExtraction.dataPoint import DocumentData

    infile = sys.argv[1]
    outfile = sys.argv[2]

    settings = config.defaultConfig
                                   
    argfe = ArgumentFeatureExtractor()
    
    #load metadata
    with open(infile) as f:
        j = json.load(f)

    training = zip(j['train_titles'], j['train_op'], j['train_pos'], j['train_neg'])
    heldout = zip(j['val_titles'], j['val_op'], j['val_pos'], j['val_neg'])

    featureLabels = {'train_features': pd.DataFrame(), 'val_features': pd.DataFrame()}
    for dataname, dataset in (('train_features', training),
                              ('val_features', heldout)):
        for count,thread in enumerate(dataset):
            if count % 10 == 0:
                print(dataname, count, len(featureLabels[dataname]))
            #for each thread, create a DocumentData object
            pos = DocumentData(thread[0], thread[1], thread[2])
            neg = DocumentData(thread[0], thread[1], thread[3])
        
            #get features for this object and add to the list
            pos_features = argfe.addFeatures(pos)
            neg_features = argfe.addFeatures(neg)
            
            #for sentences, dont forget interaction features
            #TODO
            pos_features['label'] = 1
            pos_features['index'] = len(featureLabels)
            featureLabels[dataname] = featureLabels[dataname].append(pos_features,
                                                                     ignore_index=True)
            neg_features['label'] = 0
            neg_features['index'] = len(featureLabels)            
            featureLabels[dataname] = featureLabels[dataname].append(neg_features,
                                                                     ignore_index=True)            

    train_features = featureLabels['train_features'].fillna(0).to_json()
    val_features = featureLabels['val_features'].fillna(0).to_json()
            
    #save extracted features
    with open(outfile, 'w') as f:
        json.dump({'train_features': train_features,
                   'val_features': val_features,
                   'train_labels': [1,0]*(len(featureLabels['train_features'])//2),
                   'val_labels': [1,0]*(len(featureLabels['val_features'])//2)},
                   f)


