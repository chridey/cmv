import math
import numpy as np

from altlex.featureExtraction.baseFeatureExtractor import BaseFeatureExtractor
from altlex.utils import wordUtils

from cmv.featureExtraction.altlexHandler import AltlexHandler, countValidConnectives
from cmv.featureExtraction import config

class ArgumentFeatureExtractor(BaseFeatureExtractor):
    '''features for an entire document'''
    def __init__(self,
                 settings=config.defaultConfig,
                 verbose=False,
                 cache=True):
        self.config = config.Config(settings)
        
        self.altlexHandler = AltlexHandler(**self.config.altlexSettings)
        
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
                              'framenet_altlex_sum': self.getFrameNetAltlexSum
                              }
                              #'connective_patterns'
                              #intersections
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

        self.cache = {} if cache else None
        
    def getCausalPct(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                 wordUtils.causal_markers)
        return {self.functionFeatures[self.getCausalPct]: countValidConnectives(sentences)/len(dataPoint.response)}

    def getNonCausalPct(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                 wordUtils.noncausal_markers)
        return {self.functionFeatures[self.getNonCausalPct]: countValidConnectives(sentences)/len(dataPoint.response)}
    
    def getCausalAltlexPct(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                 self.altlexHandler.causalAltlexes,
                                                 checkCache=True)
        
        return {self.functionFeatures[self.getCausalAltlexPct]: countValidConnectives(sentences)/len(dataPoint.response)}
    
    def getNonCausalAltlexPct(self, dataPoint):
         sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                  self.altlexHandler.nonCausalAltlexes)
         return {self.functionFeatures[self.getNonCausalAltlexPct]: countValidConnectives(sentences)/len(dataPoint.response)}

    def getCausalScore(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                 self.altlexHandler.causalAltlexes,
                                                 checkCache=True)
        causal, noncausal = self.altlexHandler.causalScore(sentences, empty=False)
        return {self.functionFeatures[self.getCausalScore] + '_causal' : causal/len(dataPoint.response),
                self.functionFeatures[self.getCausalScore] + '_noncausal' : noncausal/len(dataPoint.response)}
    
    #still use the lexical semantic features for interaction between title/OP and response
    #but sum over cartesian product and then L2 normalize

    def getWordNetResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        features = self.altlexHandler.getWordNetFullSentence(sentences)
        return {self.functionFeatures[self.getWordNetResponse] + '_' + k:v for (k,v) in features.items()}
        
    def getVerbNetResponse(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        features = self.altlexHandler.getVerbNetFullSentence(sentences)
        return {self.functionFeatures[self.getWordNetResponse] + '_' + k:v for (k,v) in features.items()}
    
    def getWordNetTitleResponse(self, dataPoint):
        
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        title = self.altlexHandler.getConnectiveSentences(dataPoint.title,
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
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        title = self.altlexHandler.getConnectiveSentences(dataPoint.title,
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
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        op = self.altlexHandler.getConnectiveSentences(dataPoint.originalPost,
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
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                    self.altlexHandler.causalAltlexes,
                                                    checkCache=True)
        op = self.altlexHandler.getConnectiveSentences(dataPoint.originalPost,
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
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                            self.altlexHandler.causalAltlexes,
                                                            checkCache=True)
        causal_sum,anticausal_sum = self.altlexHandler.getFrameNetResponseSum(sentences)
        return {self.functionFeatures[self.getFrameNetResponse] + '_causal': causal_sum,
                self.functionFeatures[self.getFrameNetResponse] + '_anticausal': anticausal_sum}

    
    def getFrameNetAltlexSum(self, dataPoint):
        sentences = self.altlexHandler.getConnectiveSentences(dataPoint.response,
                                                        self.altlexHandler.causalAltlexes,
                                                        checkCache=True)
        causal_sum,anticausal_sum = self.altlexHandler.getFrameNetAltlexSum(sentences)
        return {self.functionFeatures[self.getFrameNetAltlexSum] + '_causal': causal_sum,
                self.functionFeatures[self.getFrameNetAltlexSum] + '_anticausal': anticausal_sum}
    
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

    featureLabels = {'training': [], 'heldout': []}
    for dataname, dataset in (('training', training),
                              ('heldout', heldout)):
        for count,thread in enumerate(dataset):
            if count % 1 == 0:
                print(dataname, count)
            #for each thread, create a DocumentData object
            pos = DocumentData(thread[0], thread[1], thread[2])
            neg = DocumentData(thread[0], thread[1], thread[3])
        
            #get features for this object and add to the list
            pos_features = argfe.addFeatures(pos)
            neg_features = argfe.addFeatures(neg)
            
            #for sentences, dont forget interaction features
            #TODO
            
            featureLabels[dataname].append((pos_features,1))
            featureLabels[dataname].append((neg_features,0))            
            
    #save extracted features
    with open(outfile, 'w') as f:
        json.dump(featureLabels, f)
