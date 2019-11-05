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

from cmv.featureExtraction.featureExtractor import ArgumentFeatureExtractor

MAX_SENTENCE_LEN = 32
MAX_POST_LEN = 40

@DatasetReader.register("cmv")
class CMVReader(DatasetReader):
    def __init__(self,
                 data_path,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_len: int=MAX_SENTENCE_LEN,
                 max_post_len: int=MAX_POST_LEN,
                 feature_settings=None) -> None:

        self._data_path = data_path
        
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers        

        self.max_sentence_len = max_sentence_len
        self.max_post_len = max_post_len

        self._feature_settings = feature_settings

        self._feature_extractor = None
        if feature_settings is not None:
            self._feature_extractor = ArgumentFeatureExtractor(feature_settings)
        
    @overrides
    def read(self, key,
             root_response=False,
             response_only=False,
             include_title=True,
             ignore_border=False,
             ignore_quote=False,
             op_only=False,
             weakpoints_only=False):

        print('reading data from disk...')
        fopen = open
        if self._data_path.endswith('.gz'):
            fopen = gzip.open
            
        with fopen(self._data_path) as f:
            data = json.load(f)[key]

        print('processing...')                
        inverse_indices = collections.defaultdict(list)
        for label in ('pos', 'neg'):
            for index in range(len(data[label])):
                op_index = data[label + '_indices'][index]
                inverse_indices[op_index].append([index, label])

        instances = []
        print('processing...')
        for op_index in tqdm.tqdm(inverse_indices):
            original_post = [' '.join(i['words']) for i in data['op'][op_index]]
            
            title = []
            if include_title:
                title = [' '.join(i['words']) for i in data['titles'][op_index]]
            original_post = title + original_post

            op_features=None
            if self._feature_settings is not None:
                #original_post, op_features, op_doc_features = extract_features(data['op'][op_index])
                original_post, op_features = self._feature_extractor.extractSentenceFeatures(data['op'][op_index],
                                                                                             data['op'][op_index])
                op_doc_features = None
            
            if op_only:
                if len(original_post):
                    instances.append(self.text_to_instance(0,
                                                        original_post=original_post,
                                                        op_features=op_features))
                continue
                
            for index, label in inverse_indices[op_index]:
                response = []
                for i in data[label][index]:
                    if i['words'][0].lower() == 'intermediate_discussion':
                        if root_response:
                            break
                        if ignore_border:
                            continue
                    if ignore_quote and i['words'][0].lower() == 'q_u_o_t_e':
                        continue
                    response.append(' '.join(i['words']))
                    
                response_features=None
                if self._feature_settings is not None:
                    #response, response_features, response_doc_features = extract_features(data[label][index])
                    response, response_features = self._feature_extractor.extractSentenceFeatures(data[label][index],
                                                                                                  data['op'][op_index])
                    response_doc_features = None
                    
                if not len(original_post) or not len(response):
                    print('problem with original post %s and response %s' % (op_index, index))
                    continue

                weakpoints = self.get_weakpoints(data.get(label+'_quoted_text', None), index, title)
                if weakpoints_only and weakpoints is not None and weakpoints[0] == -1:
                    continue
                #print(weakpoints)
                if self._feature_settings is not None:
                    weakpoints = adjust_points(data['op'][op_index], weakpoints, [{'words':t} for t in title])
                #print(weakpoints)
                
                quoted_text = data.get('op_'+label+'_quoted_text', None)
                if quoted_text is not None:
                    quoted_text = [list(itertools.chain(*quoted_text[index]))]
                goodpoints = self.get_weakpoints(quoted_text, 0)
                #print(goodpoints)
                if self._feature_settings is not None:
                    goodpoints = adjust_points(data[label][index], goodpoints)
                #print(goodpoints)
                
                instances.append(self.text_to_instance(int(label=='pos'),
                                                       response, 
                                                       original_post=None if response_only else original_post,
                                                       weakpoints=weakpoints,
                                                       op_features=op_features,
                                                       response_features=response_features,
                                                       #op_doc_features=op_doc_features,
                                                       #response_doc_features=response_doc_features,
                                                       ))#goodpoints=goodpoints))

        return Batch(instances)

    def get_weakpoints(self, quoted_text, index, title=None):
        #data[label+'_quoted_text'][index] if label+'_quoted_text' in data and len(data[label+'_quoted_text'][index]) else [-1]
        if quoted_text is None:
            return None
        weakpoints = []
        title_len = 0 if title is None else len(title)
        for weakpoint in quoted_text[index]:
            if weakpoint + title_len < self.max_post_len:
                weakpoints.append(weakpoint + title_len)
        if len(weakpoints):
            return weakpoints
        return [-1]
            
    @overrides
    def text_to_instance(self,                         
                         label,
                         response=None,
                         original_post=None,
                         weakpoints=None,
                         op_features=None,
                         response_features=None,
                         op_doc_features=None,
                         response_doc_features=None,
                         goodpoints=None) -> Instance:

        fields: Dict[str, Field] = {}

        if original_post is not None:
            fields['original_post'] = ListField([TextField(self._tokenizer.tokenize(s)[:self.max_sentence_len],
                                                        self._token_indexers) for s in original_post[:self.max_post_len]])
            if weakpoints is not None:
                fields['weakpoints'] = ListField([IndexField(wp, fields['original_post']) for wp in weakpoints])                                

        if response is not None:
            fields['response'] = ListField([TextField(self._tokenizer.tokenize(s)[:self.max_sentence_len],
                                                    self._token_indexers) for s in response[:self.max_post_len]])

            if goodpoints is not None:
                fields['goodpoints'] = ListField([IndexField(gp, fields['response']) for gp in goodpoints])
                
        if op_features is not None:
            fields['op_features'] = ListField([ArrayField(np.array(f)) for f in op_features[:self.max_post_len]])

        if response_features is not None:
            fields['response_features'] = ListField([ArrayField(np.array(f)) for f in response_features[:self.max_post_len]])

        if op_doc_features is not None:
            fields['op_doc_features'] = ArrayField(np.array(op_doc_features))

        if response_doc_features is not None:
            fields['response_doc_features'] = ArrayField(np.array(response_doc_features))
                        
        fields['label'] = LabelField(label, skip_indexing=True)

        return Instance(fields)

#if we are using structural features, there are fewer sentences bc we don't have quote/url/intermediate_discussion    
def adjust_points(metadata, points, title=None):
    if points is None or (len(points) and points[0] == -1):
        return points
    point_index = 0
    sentence_index = 0
    final_points = []

    if title is None:
        title = []
    #print(len(title), len(metadata))
    for index,sentence in enumerate(metadata + title):
        if not len(sentence['words']) or sentence['words'][0].lower() == 'intermediate_discussion' or sentence['words'][0].lower() == 'q_u_o_t_e' or sentence['words'][0].lower() == 'u_r_l':
            continue
        #print(index, sentence_index, point_index)        
        if index >= points[point_index]:
            final_points.append(points[point_index] - (index-sentence_index) - len(title))
            point_index += 1
            if point_index >= len(points):
                break        
        sentence_index += 1
        
    return final_points

