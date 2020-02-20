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
from cmv.preprocessing.cmvReader import CMVReader

@DatasetReader.register("cmv_fusion")
class CMVFusionReader(CMVReader):

    @overrides
    def read(self, data_path):
        with open(data_path) as f:
            for i,line in enumerate(f):
                datum = json.loads(line)
                for j,instance in self.iter_instance(datum):
                    yield i,j,datum,instance
                    
    def get_op(self, op):
        op_paragraphs = [self.embedParagraph([i]) for i in op]

        original_post = ['a' for i in op]
        if not len(original_post):
            original_post = ['a']
            op_paragraphs = [np.zeros(300)]
            
        return op_paragraphs, original_post

    def get_post(self, post):
        response = []
        paragraphs = []
        paragraph_index = 0
        paragraph = []
        for sentence in post:
            if 'Q_U_O_T_E' in sentence['original'][0] or 'U_R_L' in sentence['original'][0]:
                continue
            if sentence['paragraph_index'] != paragraph_index:
                if len(paragraph):
                    response.append(paragraph[0]['words'][0])
                    paragraphs.append(self.embedParagraph(paragraph))
                    paragraph_index = sentence['paragraph_index']
                    paragraph = []
            paragraph.append(sentence)
        if len(paragraph):
            response.append(paragraph[0]['words'][0])
            paragraphs.append(self.embedParagraph(paragraph))

        if not len(paragraphs):
            print('problem with post')
            raise Exception

        return paragraphs, response        
                        
    def iter_instance(self, datum):
        #this is really some number of candidates
        
        original_post = None                
        op_features=None
        op_paragraphs = None
                
        if not self._response_only:
            op_paragraphs, original_post = self.get_op(datum['parent']['data'])
                                            
        response_features=None
        instances = []
        metadata = []
        for index,(ordered_paragraphs,score,shell) in enumerate(zip(datum['order'], datum['biases'], datum['shell'])):
            if ordered_paragraphs is None:
                continue
            paragraphs, response = self.get_post(ordered_paragraphs['data'])
            global_features = [score]

            instance = self.text_to_instance(int(0), response,
                              original_post=None if self._response_only else original_post,
                              op_features=op_features,
                              response_features=response_features,
                              global_features=global_features,
                              paragraphs=paragraphs,
                              op_paragraphs=op_paragraphs,
                              )
            yield index,instance
                

            
