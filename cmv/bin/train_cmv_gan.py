from __future__ import print_function

import os
import argparse
import json
import time
import collections

from copy import deepcopy

import numpy as np

import torch

from allennlp.nn import util
from allennlp.data import Vocabulary, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.common import Params
from allennlp.data.dataset import Batch
from allennlp.models import Model, archive_model, archival
from allennlp.training import Trainer

from cmv.preprocessing.cmvReader import CMVReader
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvExtractor import CMVExtractor
from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvDiscriminator import CMVDiscriminator
from cmv.rnn.cmvActorCritic import CMVActorCritic
from cmv.rnn.cmvGeneratorTrainer import CMVGeneratorTrainer
from cmv.rnn.cmvDiscriminatorTrainer import CMVDiscriminatorTrainer
from cmv.rnn.cmvActorCriticTrainer import CMVActorCriticTrainer

from cmv.rnn.ganTrainer import GANTrainer

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

    logger.info('Reading training data...')
    
    train_data = dataset_reader.read('train', **data_params)

    #train_data_response_only_for_vocab = dataset_reader.read('train', response_only=True)
    #all_datasets = [train_data_response_only_for_vocab]
    all_datasets = [train_data]
    datasets_in_vocab = ['train'] #_response_only_for_vocab']

    if use_validation_data:
        logger.info('Reading validation data...')
        data_params['weakpoints_only'] = False
        validation_data = dataset_reader.read('val', **data_params)
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
    
    iterator = DataIterator.from_params(params.pop('iterator'))

    cmv_predictor_params = params.pop('cmv_predictor')
    predictor_pretrained_params = cmv_predictor_params.pop('predictor_pretrained_params', None)
    cmv_predictor = Model.from_params(params=cmv_predictor_params, vocab=vocab)
    model_state = torch.load(predictor_pretrained_params['filename'],
                             map_location=util.device_mapping(predictor_pretrained_params['cuda_device']))
    cmv_predictor.load_state_dict(model_state)
    
    if params.pop('shared_embedder', False):
        print('using shared embedder')
        document_embedder = HierarchicalDocumentEmbedder(vocab, cmv_predictor._response_embedder,
                                                         cmv_predictor._response_word_attention,
                                                         cmv_predictor._response_encoder)
    else:
        document_embedder = Model.from_params(params=params.pop('document_embedder'), vocab=vocab)
    cmv_extractor = Model.from_params(params=params.pop('cmv_extractor'))
    cmv_discriminator = Model.from_params(params=params.pop('cmv_discriminator'))
    
    cmv_actor_critic_params = params.pop('cmv_actor_critic', None)
    cmv_actor_critic = None
    if cmv_actor_critic_params is not None:
        cmv_actor_critic = Model.from_params(params=cmv_actor_critic_params)
        
    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer", None)
    if trainer_params is not None:
        if cuda_device is not None:
            trainer_params["cuda_device"] = cuda_device
        trainer = Trainer.from_params(cmv_predictor,
                                      serialization_dir,
                                      iterator,
                                      train_data,
                                      validation_data,
                                      trainer_params)

    compress_response=params.pop('compress_response', False)
        
    generator_iterator = DataIterator.from_params(params.pop('generator_iterator'))    
    cmv_actor_critic_trainer_params = params.pop('actor_critic_trainer', None)
    if cmv_actor_critic_trainer_params is not None:
        cmv_actor_critic_pretrainer = CMVActorCriticTrainer(document_embedder, 
                                                            cmv_predictor, cmv_extractor,
                                                            cmv_actor_critic,
                                                            cmv_actor_critic_trainer_params.pop('train_predictor', False),
                                                            cmv_actor_critic_trainer_params.pop('train_fake_predictor', False),
                                                            compress_response)
                                                            
        cmv_actor_critic_serialization_dir = os.path.join(serialization_dir, 'actor_critic')
                        
        cmv_actor_critic_trainer = Trainer.from_params(cmv_actor_critic_pretrainer,
                                    cmv_actor_critic_serialization_dir,
                                    generator_iterator,
                                    train_data,
                                    validation_data,
                                    cmv_actor_critic_trainer_params)
    else:
        ac_pretrained_params = params.pop('pretrained_actor_critic', None)
        if ac_pretrained_params is not None:
            cmv_actor_critic_pretrainer = CMVActorCriticTrainer(document_embedder, 
                                                                cmv_predictor, cmv_extractor,
                                                                None)
            model_state = torch.load(ac_pretrained_params['filename'],
                                     map_location=util.device_mapping(ac_pretrained_params['cuda_device']))
            cmv_actor_critic_pretrainer.load_state_dict(model_state)
            document_embedder = cmv_actor_critic_pretrainer._document_embedder
            cmv_predictor = cmv_actor_critic_pretrainer._cmv_predictor
            cmv_extractor = cmv_actor_critic_pretrainer._cmv_extractor

    generator = CMVGeneratorTrainer(document_embedder, cmv_predictor, cmv_extractor,
                            cmv_discriminator, cmv_actor_critic,
                            update_extractor = True, #cmv_actor_critic_trainer_params is None,
                            update_gold_extractor = False, #True,
                            compress_response=compress_response) #False)
    
    discriminator = CMVDiscriminatorTrainer(document_embedder, cmv_predictor,
                                            cmv_extractor, cmv_discriminator,
                                            compress_response)

    generator_serialization_dir = os.path.join(serialization_dir, 'generator')
    os.makedirs(generator_serialization_dir, exist_ok=True)
    generator_trainer = GANTrainer.from_params(generator,
                                    generator_serialization_dir,
                                    generator_iterator,
                                    train_data,
                                    validation_data,
                                    params.pop('generator_trainer'))

    discriminator_serialization_dir = os.path.join(serialization_dir, 'discriminator')
    os.makedirs(discriminator_serialization_dir, exist_ok=True)
    discriminator_trainer = GANTrainer.from_params(discriminator,
                                    discriminator_serialization_dir,
                                    iterator,
                                    train_data,
                                    validation_data,
                                    params.pop('discriminator_trainer'))
        
    #first train predictor for N steps
    if trainer_params is not None:
        trainer._num_epochs = 5 #hacky
        trainer.train()
    
    #TODO? then train actor critic for M steps
    #if we are using separate predictors, use the full CMV to train the extractor based on maximizing persuasiveness prediction
    if cmv_actor_critic_trainer_params is not None:
        cmv_actor_critic_trainer.train()
        
    #then alternate training between discriminator and generator for E epochs
    generator_trainer._num_epochs = 1 #hacky
    discriminator_trainer._num_epochs = 1 #hacky
    gan_epochs = params.pop("gan_epochs")
    for i in range(gan_epochs):
        discriminator_trainer.train()
        generator_trainer.train()        
        discriminator_trainer._num_epochs += 1 #very hacky
        generator_trainer._num_epochs += 1 #also hacky

        #if cmv_actor_critic_trainer_params is not None:
        #    cmv_actor_critic_trainer._num_epochs += 1
        #    cmv_actor_critic_trainer.train()
        
    # Now tar up results
    archive_model(serialization_dir)
    archive_model(generator_serialization_dir)
    archive_model(discriminator_serialization_dir)        

    return generator
                                
        
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
    
