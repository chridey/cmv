import os
import json
import collections
import math

from sklearn.externals import joblib

from altlex.featureExtraction.featureExtractor import FeatureExtractor
from altlex.featureExtraction import config
from altlex.featureExtraction.dataPoint import makeDataPoint, makeDataPointsFromAltlexes

def countValidConnectives(sentences):
    return 1.*sum(i.altlexLength > 0 for i in sentences)

class AltlexHandler:
    def __init__(self,
                 featureSettings=None,
                 classifierFile=None,
                 altlexFile=None,
                 cache=True,
                 verbose=False):

        if featureSettings is None:
            featureSettings = config.defaultConfig

        self.featureExtractor = FeatureExtractor(featureSettings,
                                                 verbose)

        self._classifierFile = classifierFile
        self._classifier = None #best classifier trained on wikipedia, load from settings
        self._vectorizer = None
        
        self.altlexFile = altlexFile
        self._causalAltlexes = None #causal altlexes, load from settings
        self._nonCausalAltlexes = None
        
        self.cache = {} if cache else None  #for sentence features/metadata to prevent it from being recalculated many times

    def loadAltlexes(self):
        if self.altlexFile is None:
            self.altlexFile = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),
                                                            '..'), 'config'), 'altlexes.json')
        with open(self.altlexFile) as f:
            altlexes = json.load(f)

        self._causalAltlexes = altlexes['causal']
        self._nonCausalAltlexes = altlexes['noncausal']

    @property
    def classifierFile(self):
        if self._classifierFile is None:
            self._classifierFile = os.path.join(os.path.join(os.path.join(os.path.dirname(__file__),
                                                            '..'), 'config'),
                                                            'full_plus_sgd_st1_inter1_unbalanced_combined_bootstrap.1')
                                                            #'altlexes.classifier')
        return self._classifierFile
    
    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = joblib.load(self.classifierFile)
        return self._classifier
    
    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.classifierFile + '.vectorizer')
        return self._vectorizer

    @property
    def causalAltlexes(self):
        if self._causalAltlexes is None:
            self.loadAltlexes()
        return self._causalAltlexes          

    @property
    def nonCausalAltlexes(self):
        if self._nonCausalAltlexes is None:
            self.loadAltlexes()
        return self._nonCausalAltlexes          
        
    def getConnectiveSentences(self, metadataList, connectives,
                               checkCache=False):
        if checkCache and self.cache is not None and 'data' in self.cache:
            return self.cache['data']
        
        connectiveSentences = []
        for sentence in metadataList:
            connectiveSentences.extend(makeDataPointsFromAltlexes(sentence, connectives, True))

        if checkCache and self.cache is not None:
            self.cache['data'] = connectiveSentences
            
        return connectiveSentences

    def getCausalConnectiveSentences(self, metadataList, checkCache=True):
        return self.getConnectiveSentences(metadataList, self.causalAltlexes, checkCache)

    def addFeatures(self, metadataList):
        if self.cache is not None and 'features' in self.cache:
            return self.cache['features']

        features = []
        for dataPoint in metadataList:
            features.append(self.featureExtractor.addFeatures(dataPoint))
        
        if self.cache is not None:
            self.cache['features'] = features

        return features
    
    def causalScore(self, metadataList, empty=False):
        validIndices = {i for i in range(len(metadataList)) if metadataList[i].altlexLength}

        #extract features for each sentence and get the decision_function
        features = [f for (i,f) in enumerate(self.addFeatures(metadataList)) if i in validIndices]

        try:
            features_transformed = self.vectorizer.transform(features)
        except ValueError:
            print("problem with feature transformer")
            return 0.,0.
        
        predictions = self.classifier.decision_function(features_transformed)
        
        #sum or accumulate the scores over each sentence
        return sum(i for i in predictions if i > 0), sum(i for i in predictions if i < 0)

    def causalPredictions(self, metadataList):
        #extract features for each sentence and get the decision_function
        features = [f for (i,f) in enumerate(self.addFeatures(metadataList))]
        try:
            features_transformed = self.vectorizer.transform(features)
        except ValueError:
            print("problem with feature transformer")
            return [0]*len(metadataList)
        
        return self.classifier.predict(features_transformed)

    def getSemanticsFullSentence(self, metadataList, resourceType, normalize=True):
        validIndices = {i for i in range(len(metadataList)) if not metadataList[i].altlexLength}
        featureList = [f for (i,f) in enumerate(self.addFeatures(metadataList)) if i in validIndices]

        featureCount = collections.Counter(feature for features in featureList for feature in features if 'arguments_' + resourceType in feature or 'head_word_' + resourceType in feature)
        l2 = 1
        if normalize:
            l2 = math.sqrt(sum(i**2 for i in featureCount.values()))

        return {k:v/l2 for (k,v) in featureCount.items()}

    def getWordNetFullSentence(self, metadataList, normalize=True):
        return self.getSemanticsFullSentence(metadataList, 'cat', normalize)
        
    def getVerbNetFullSentence(self, metadataList, normalize=True):
        return self.getSemanticsFullSentence(metadataList, 'verbnet', normalize)

    def getFrameNetSum(self, metadataList, fragment):
        if fragment == 'altlex':
            validIndices = {i for i in range(len(metadataList)) if metadataList[i].altlexLength}
        else:
            validIndices = {i for i in range(len(metadataList)) if not metadataList[i].altlexLength}
        featureList = [f for (i,f) in enumerate(self.addFeatures(metadataList)) if i in validIndices]
        causal_sum = 0
        anticausal_sum = 0
        for features in featureList:
            causal_sum += features.get('framenet_{}_causal'.format(fragment), 0)
            anticausal_sum += features.get('framenet_{}_anticausal'.format(fragment), 0)
        return causal_sum, anticausal_sum

    def getFrameNetAltlexSum(self, metadataList):
        return self.getFrameNetSum(metadataList, 'altlex')
    
    def getFrameNetResponseSum(self, metadataList):
        return self.getFrameNetSum(metadataList, 'curr_post')
    
    def clearCache(self):
        if self.cache is not None:
            self.cache = {}
