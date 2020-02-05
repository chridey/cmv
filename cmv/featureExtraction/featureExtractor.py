import pandas as pd
import nltk
import numpy as np
from sklearn.externals import joblib

from cmv.preprocessing.thread import Post,Thread

def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

class ArgumentFeatureExtractor:
    '''features for an entire document or sentence'''

    emotion_embeddings_file = '/proj/nlp/users/chidey/cmv/ewe_uni.txt'
    
    def __init__(self,
                 settings=None,
                 verbose=False):

        if settings is not None:
            self.settings = settings
        else:
            self.settings = {'featureSettings': {'interplay': True,
                                                 'emotion': True,
                                                 'structure': True,
                                                 'tfidf': False}}
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
                
        self.validFeatures = {'interplay': self.getInterplay,
                              'emotion': self.getEmotions,
                              'structure': self.getStructure,
                              'tfidf': self.getTfidf,
                              }
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

        self._emotionEmbeddings = None
        self._emotionEmbeddingsDim = None

        self._tfidf_vectorizer = None
        
    @property
    def tfidf_vectorizer(self):
        if self._tfidf_vectorizer is None:
            if 'vectorizer' in self.settings:
                self._tfidf_vectorizer = joblib.load(self.settings['vectorizer'])
                
        return self._tfidf_vectorizer
    
    def getTfidf(self, dataPoint, *args):
        pass
        #words = ' '.join(dataPoint.response.getAllWords(True))
        #self.tfidf_vectorizer.fit([words])
        
    @property
    def emotionEmbeddings(self):
        if self._emotionEmbeddings is None:
            self._emotionEmbeddings = {}
            with open(self.emotion_embeddings_file) as f:
                for line in f:
                    word, *embedding = line.split()
                    self._emotionEmbeddings[word] = np.array(list(map(float,embedding)))
                    if self._emotionEmbeddingsDim is None:
                        self._emotionEmbeddingsDim = self._emotionEmbeddings[word].shape[0]
                    assert(self._emotionEmbeddingsDim == self._emotionEmbeddings[word].shape[0])
                    
        return self._emotionEmbeddings
    
    def getEmotions(self, dataPoint, *args):
        words = dataPoint.response.getAllWords(True)
        embedded_sentence = np.stack([self.emotionEmbeddings.get(word,
                                                                 np.zeros(self._emotionEmbeddingsDim)) for word in words])
        
        features = np.concatenate([embedded_sentence.max(axis=0),
                                  embedded_sentence.min(axis=0),
                                  embedded_sentence.mean(axis=0)])
        
        keys = ['EMAX' + str(i) for i in range(self._emotionEmbeddingsDim)] + ['EMIN' + str(i) for i in range(self._emotionEmbeddingsDim)] + ['EMEAN' + str(i) for i in range(self._emotionEmbeddingsDim)]
        return dict(zip(keys, features))
        
    def getInterplay(self, dataPoint, *args):
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
        return dict(zip(keys, all_interplay + stop_interplay + content_interplay))

    def getStructure(self, dataPoint, structureFeatures):
        features = structureFeatures.getFeatures()
        keys = ['STRUC' + str(i) for i in range(len(features))]
        return dict(zip(keys, features))
    
    def addFeatures(self, dataPoint, featureSettings=None):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']

        if 'structure' in featureSettings:
            featureSettings['structure'] = False
                        
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint, None))
        return features
        
        return features

    def extractSentenceFeatures(self, response, post=None, featureSettings=None):
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']
            
        features = []
        structureFeatures = StructureFeatures(response)

        thread = []
        for sentence in response:
            if not len(sentence['words']) or sentence['words'][0].lower() in ('intermediate_discussion',
                                                                              'q_u_o_t_e',
                                                                              'u_r_l'):
                continue

            dataPoint = Thread([sentence], post)
            sentence_features = []
            for featureName in featureSettings:
                assert(featureName in self.validFeatures)
                if featureSettings[featureName]:
                    sentence_features.extend(self.validFeatures[featureName](dataPoint,
                                                                             structureFeatures).values())

            features.append(sentence_features)
            thread.append(' '.join(sentence['words']))
                                
        return thread, features
        
class StructureFeatures:
    def __init__(self, response):
        #for each sentence, extract distance from start/end of paragraph/post/thread                                                                                                                                                 
        #also extract paragraph distance from start/end of post/thread and post distance from start/end of thread                                                                                                                    
        #distance from previous/next quote/url (only do this within a single post or across posts? i think only within a single post)                                                                                                
        #0 if N/A                                                                                                                                                                                                                    
        #also is_title feature                                                                                                                                                                                                       

        #keep track of breaks, the location of each of these                                                                                                                                                                         
        self.paragraph_breaks = [0]
        self.post_breaks = []
        self.quote_breaks = []
        self.url_breaks = []
        #for each quote, url, and paragraph, what post does it belong to
        self.paragraph_post_breaks = [0]

        sentence_index = 0
        paragraph_index = 0
        self.thread_len = 0
        for sentence in response:
            if not len(sentence['words']):
                continue
            if sentence['words'][0].lower() == 'intermediate_discussion':
                self.post_breaks.append(sentence_index)
                self.paragraph_post_breaks.append(len(self.paragraph_breaks))
            elif sentence['words'][0].lower() == 'q_u_o_t_e':
                self.quote_breaks.append(sentence_index)
            elif sentence['words'][0].lower() == 'u_r_l':
                self.url_breaks.append(sentence_index)
            else:
                self.thread_len += 1
                if sentence['paragraph_index'] != paragraph_index:
                    self.paragraph_breaks.append(sentence_index)
                    self.paragraph_index = sentence['paragraph_index']
                sentence_index += 1

        self.paragraph_breaks.append(self.thread_len)
        self.post_breaks.insert(0,0)
        self.post_breaks.append(self.thread_len)
        self.paragraph_post_breaks.append(len(self.paragraph_breaks))

        #print(paragraph_breaks, post_breaks, quote_breaks, url_breaks, paragraph_post_breaks)
        self.paragraph_marker = 0
        self.post_marker = 0
        self.quote_marker = 0
        self.url_marker = 0
        self.i = 0
        
    def getFeatures(self):
        d_from_start_of_paragraph = self.i-self.paragraph_breaks[self.paragraph_marker] + 1
        d_from_start_of_post = self.i-self.post_breaks[self.post_marker] + 1
        d_from_start_of_thread = self.i + 1

        d_from_end_of_paragraph = self.paragraph_breaks[self.paragraph_marker+1]-self.i
        d_from_end_of_post = self.post_breaks[self.post_marker+1]-self.i
        d_from_end_of_thread = self.thread_len-self.i

        paragraph_d_from_start_of_post = self.paragraph_marker-self.paragraph_post_breaks[self.post_marker] + 1
        paragraph_d_from_end_of_post = self.paragraph_post_breaks[self.post_marker+1]-self.paragraph_marker

        d_from_previous_quote = 0
        if self.quote_marker != 0 and self.quote_marker-1 < len(self.quote_breaks) and self.quote_breaks[self.quote_marker-1] > self.post_breaks[self.post_marker]:
            d_from_previous_quote = self.i-self.quote_breaks[self.quote_marker-1] + 1
        d_from_next_quote = 0
        if self.quote_marker < len(self.quote_breaks) and self.quote_breaks[self.quote_marker] < self.post_breaks[self.post_marker+1]:
            d_from_next_quote = self.quote_breaks[self.quote_marker]-self.i

        d_from_previous_url = 0
        if self.url_marker != 0 and self.url_marker-1 < len(self.url_breaks) and self.url_breaks[self.url_marker-1] > self.post_breaks[self.post_marker]:
            d_from_previous_url = self.i-self.url_breaks[self.url_marker-1] + 1
        d_from_next_url = 0
        if self.url_marker < len(self.url_breaks) and self.url_breaks[self.url_marker] < self.post_breaks[self.post_marker+1]:
            d_from_next_url = self.url_breaks[self.url_marker]-self.i

        features = [d_from_start_of_paragraph, d_from_start_of_post, d_from_start_of_thread,
                    d_from_end_of_paragraph, d_from_end_of_post, d_from_end_of_thread,
                    self.paragraph_marker+1, len(self.paragraph_breaks)-1-self.paragraph_marker,
                    paragraph_d_from_start_of_post, paragraph_d_from_end_of_post,
                    self.post_marker+1, len(self.post_breaks)-1-self.post_marker,
                    d_from_previous_quote, d_from_next_quote,
                    d_from_previous_url, d_from_next_url]

        if self.paragraph_breaks[self.paragraph_marker+1] == self.i+1:
            self.paragraph_marker += 1
        if self.post_breaks[self.post_marker+1] == self.i+1:
            self.post_marker += 1

        if self.quote_marker < len(self.quote_breaks) and self.quote_breaks[self.quote_marker] == self.i+1:
            self.quote_marker += 1
        if self.url_marker < len(self.url_breaks) and self.url_breaks[self.url_marker] == self.i+1:
            self.url_marker += 1

        self.i += 1
        
        return features
