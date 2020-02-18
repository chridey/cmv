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

from cmv.preprocessing.cmvReader import CMVReader
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
    dataset_reader = CMVReader.from_params(ds_params)
        
    model = archive.model
    model.eval()

    validation_data = dataset_reader.read(**data_params['val'])
    
    val_iterator = DataIterator.from_params(params['iterator'])
    vocab = model.vocab
    val_iterator.index_with(vocab)
                
    val_generator = val_iterator(validation_data,
                                 num_epochs=1,
                                 shuffle=False)

    #model._cmv_extractor._compression_rate = args.compression_rate
    for batch in val_generator:
        output = model(**batch)
                
        #print(model._cmv_predictor.get_metrics(reset=True))
    print(model.get_metrics())        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('archive_file')

    parser.add_argument("--cuda-device", type=int, default=-1, help='id of GPU to use (if any)')

    args = parser.parse_args()

    eval_model(args)    
