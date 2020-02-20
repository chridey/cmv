from __future__ import print_function

import os
import argparse
import json
import time
import collections

from copy import deepcopy

import numpy as np

from allennlp.data import Vocabulary, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.common import Params
from allennlp.data.dataset import Batch
from allennlp.models import Model, archive_model, archival, load_archive
from allennlp.training import Trainer
from allennlp.nn import util as nn_util

from cmv.preprocessing.cmvFusionReader import CMVFusionReader
from cmv.rnn.cmvCoherencePredictor import CMVCoherencePredictor

import argparse
import logging
import sys
import json
import os

logger = logging.getLogger(__name__)  

def eval_model(args):

    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)

    params = archive.config

    ds_params = params.pop('dataset_reader', {})
    data_params = ds_params.pop('data', {})
    dataset_reader = CMVFusionReader.from_params(ds_params)
        
    model = archive.model
    model.eval()

    scores = collections.defaultdict(dict)
    data = {}    
    for i,j,datum,instance in dataset_reader.read(args.data_path):
        output = model.forward_on_instance(instance)
        data[i] = datum
        scores[i][j] = output['label_probs']
        
    '''
    validation_iter = dataset_reader.read(args.data_path)
    metadata = []
    validation_data = []
    for i,metadatum,instance in validation_iter:
        validation_data.append(instance)
        metadata.append([i,metadatum])
        
    val_iterator = BasicIterator()
    vocab = model.vocab
    val_iterator.index_with(vocab)
                
    val_generator = val_iterator(validation_data,
                                 num_epochs=1,
                                 shuffle=False)

    for batch in val_generator:
        #batch = nn_util.move_to_device(batch, args.cuda_device)
        output = model(**batch)
                
        #TODO: need to get the best score out of all the candidates
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file')
    parser.add_argument('data_path')    

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')

    args = parser.parse_args()

    eval_model(args)    
