from typing import Dict, Optional, List, Any
from overrides import overrides

import gzip
import json
import logging
import collections
import itertools

import numpy as np

import tqdm

from allennlp.common import Params
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField, IndexField, ArrayField

from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)

from cmv.preprocessing.thread import Thread,Post
from cmv.featureExtraction.featureExtractor import ArgumentFeatureExtractor

MAX_SENTENCE_LEN = 32
MAX_POST_LEN = 40
MAX_NUM_PARAGRAPHS = 16

@DatasetReader.register("cmv")
class CMVReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_len: int=MAX_SENTENCE_LEN,
                 max_post_len: int=MAX_POST_LEN,
                 max_num_paragraphs: int=MAX_NUM_PARAGRAPHS,
                 feature_settings=None,
                 coherence_settings=None,
                 response_only=True) -> None:

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers        

        self.max_sentence_len = max_sentence_len
        self.max_post_len = max_post_len
        self.max_num_paragraphs = max_num_paragraphs

        self._feature_settings = feature_settings

        self._feature_extractor = None
        if feature_settings is not None:
            self._feature_extractor = ArgumentFeatureExtractor(feature_settings)

        if 'best_features' in feature_settings:
            self._best_features = set(json.load(open(feature_settings['best_features'])))
            
        self._coherence_embedders = None
        if coherence_settings is not None:
            self._coherence_embedders = []
            if 'argument_words' in coherence_settings:
                argument_words = set(json.load(open(coherence_settings['argument_words'])))
            for coherence_setting in coherence_settings['embedders']:
                coherence_setting['argument_words'] = argument_words
                settings = dict(featureSettings=dict(coherence=dict(coherence_setting)))
                self._coherence_embedders.append(ArgumentFeatureExtractor(settings))

        self._response_only = response_only
        
    def embedParagraph(self, paragraph):
        embedding = []
        for embedder in self._coherence_embedders:
            embedding.append(embedder.getCoherenceEmbedding(Post(paragraph)))
        return np.concatenate(embedding)
                
    @overrides
    def read(self, data_path,
             feature_path=None,
             include_title=False):

        features = []
        if feature_path is not None:
            with open(feature_path) as f:
                for line in f:
                    features.append(json.loads(line))

        print('processing...')                    
        instances = []
        with open(data_path) as f:
            for index,line in enumerate(tqdm.tqdm(f)):
                datum = json.loads(line)

                global_features = None
                if not len(features):
                    if self._feature_extractor is not None:
                        global_features = self._feature_extractor.addFeatures(Thread(datum['comments'][0], datum['op']))
                    
                else:
                    global_features = features[index]
                    #label = global_features['label']
                label = datum['label']
                global_features = [global_features[i] for i in global_features if i in self._best_features]
                
                #TODO
                original_post = None                
                op_features=None
                op_paragraphs = None
                
                #TODO
                if not self._response_only:
                    op_paragraphs = [self.embedParagraph([i]) for i in datum['op']['data']]
                    
                    
                    original_post = ['a' for i in datum['op']['data']]
                    if not len(original_post):
                        original_post = ['a']
                        op_paragraphs = [np.zeros(300)]
                        
                    title = []
                    if include_title:
                        title = [' '.join(i['words']) for i in datum['title']['data']]
                        op_paragraphs = [self.embedParagraph([i]) for i in datum['title']['data']] + op_paragraphs
                    original_post = title + original_post                
                    
                #TODO
                response_features=None
                response = []
                
                paragraphs = []
                paragraph_index = 0
                paragraph = []
                for sentence in datum['comments'][0]['data']:
                    if 'Q_U_O_T_E' in sentence['original'][0] or 'U_R_L' in sentence['original'][0]:
                        continue
                    if sentence['paragraph_index'] != paragraph_index:
                        if len(paragraph):
                            #TODO
                            response.append(paragraph[0]['words'][0])
                            paragraphs.append(self.embedParagraph(paragraph))
                            paragraph_index = sentence['paragraph_index']
                            paragraph = []
                    paragraph.append(sentence)
                if len(paragraph):
                    response.append(paragraph[0]['words'][0])
                    paragraphs.append(self.embedParagraph(paragraph))

                if not len(paragraphs):
                    print('problem with index '.format(index))
                    continue
                    
                instances.append(self.text_to_instance(int(label),
                                                       response, 
                                                       original_post=None if self._response_only else original_post,
                                                       op_features=op_features,
                                                       response_features=response_features,
                                                       global_features=global_features,
                                                       paragraphs=paragraphs,
                                                       op_paragraphs=op_paragraphs,
                                                       ))
                
        return Batch(instances)
            
    @overrides
    def text_to_instance(self,                         
                         label,
                         response=None,
                         original_post=None,
                         op_features=None,
                         response_features=None,
                         global_features=None,
                         paragraphs=None,
                         op_paragraphs=None) -> Instance:

        fields: Dict[str, Field] = {}

        if original_post is not None:
            fields['original_post'] = ListField([TextField(self._tokenizer.tokenize(s)[:self.max_sentence_len],
                                                        self._token_indexers) for s in original_post[:self.max_num_paragraphs]])

        if response is not None:
            fields['response'] = ListField([TextField(self._tokenizer.tokenize(s)[:self.max_sentence_len],
                                                    self._token_indexers) for s in response[:self.max_num_paragraphs]])

                
        if op_features is not None:
            fields['op_features'] = ListField([ArrayField(np.array(f)) for f in op_features[:self.max_num_paragraphs]])

        if response_features is not None:
            fields['response_features'] = ListField([ArrayField(np.array(f)) for f in response_features[:self.max_num_paragraphs]])

        if global_features is not None:
            fields['global_features'] = ArrayField(np.array(global_features))

        if paragraphs is not None:
            fields['paragraphs'] = ListField([ArrayField(np.array(f)) for f in paragraphs[:self.max_num_paragraphs]])
            
        if op_paragraphs is not None:
            fields['op_paragraphs'] = ListField([ArrayField(np.array(f)) for f in op_paragraphs[:self.max_num_paragraphs]])
            
        fields['label'] = LabelField(label, skip_indexing=True)

        return Instance(fields)
