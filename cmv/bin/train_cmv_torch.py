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
from allennlp.models import Model, archive_model, archival
from allennlp.training import Trainer

from cmv.preprocessing.cmvReader import CMVReader
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvMultiChannelPredictor import CMVMultiChannelPredictor
#from cmv.rnn.cmvExtractor import CMVExtractor

import argparse
import logging
import sys
import json
import os

logger = logging.getLogger(__name__)  

def train_model(data_path, params, serialization_dir, cuda_device=-1, use_validation_data=True):

    os.makedirs(serialization_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    
    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)
                            
    ds_params = params.pop('dataset_reader', {})
    data_params = ds_params.pop('data', {})
    dataset_reader = CMVReader.from_params(ds_params)
    '''
    dataset_reader = CMVReader(data_path,
                               tokenizer=Tokenizer.from_params(ds_params.pop('tokenizer', {})),
                               token_indexers=TokenIndexer.from_params(ds_params.pop('token_indexers', {})))
    '''

    logger.info('Reading training data...')
    
    train_data = dataset_reader.read('train',
                                     **data_params)

    #train_data_response_only_for_vocab = dataset_reader.read('train', response_only=True)
    #train_data_op_only_for_vocab = dataset_reader.read('train', op_only=True)
    #all_datasets = [train_data_response_only_for_vocab, train_data_op_only_for_vocab]
    all_datasets = [train_data]
    datasets_in_vocab = ['train'] #_response_only_for_vocab']

    if use_validation_data:
        logger.info('Reading validation data...')
        validation_data = dataset_reader.read('val',
                                              **data_params)
        all_datasets.append(validation_data)
        datasets_in_vocab.append('val')
    else:
        validation_data = None

    logger.info('Creating a vocabulary using %s data.', ', '.join(datasets_in_vocab))
    vocab_params = params.pop('vocabulary', {})
    dataset = None
    if 'directory_path' not in vocab_params:
        dataset = Batch([instance for dataset in all_datasets
                            for instance in dataset.instances])

    vocab = Vocabulary.from_params(vocab_params,
                                   dataset)
    vocab.save_to_files(os.path.join(serialization_dir, 'vocabulary'))
    
    model = Model.from_params(params=params.pop('model'), vocab=vocab)
    iterator = DataIterator.from_params(params.pop('iterator'))

    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    if cuda_device is not None:
        trainer_params["cuda_device"] = cuda_device
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)

    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model
                                
        
parser = argparse.ArgumentParser()

parser.add_argument('data_path')
parser.add_argument('param_path',
                    type=str,
                    help='path to parameter file describing the model to be trained')
        
parser.add_argument("logdir",type=str)

parser.add_argument("--cuda-device", type=int, default=None, help='id of GPU to use (if any)')

args = parser.parse_args()

params = Params.from_file(args.param_path)

train_model(args.data_path, params, args.logdir, args.cuda_device)
    
