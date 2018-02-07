from __future__ import print_function

import os
import argparse
import json
import time
import collections

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cmv.rnn.persuasiveInfluenceClassifier import PersuasiveInfluenceClassifier, load
from cmv.rnn.vocab import build_vocab
import cmv.rnn.utils as ut

def combine_data(metadata):
    new_metadata = collections.defaultdict(lambda: collections.defaultdict(list))
    for split in ('train', 'val', 'test'):
        if split not in metadata:
            print('WARNING: {} not in metadata'.format(split))
            continue
        for name in ('pos', 'neg'):
            for index, post in enumerate(metadata[split][name]):
                if name+'_indices' in metadata[split]:
                    op_index = metadata[split][name+'_indices'][index]
                    op = metadata[split]['op'][op_index]
                    if 'titles' in metadata[split]:
                        title = metadata[split]['titles'][op_index]
                    else:
                        title = []
                else:
                    op = []
                    title = []
                new_metadata[split]['op'].append(op)
                new_metadata[split]['titles'].append(title)
                new_metadata[split]['rr'].append(post)
                new_metadata[split]['labels'].append(name=='pos')
                
    return new_metadata

def prepare(data, vocab, biases, args):

    embeddings = ut.prepare_embeddings(data['embeddings'], vocab)    
    data = combine_data(data)
    
    print('building data...')
    inputs = collections.defaultdict(list)
    for split in data:
        for name in ('rr', 'op'):
            if name == 'op' and not args.op:
                continue
            inputs[split].extend(ut.prepare_data(data[split][name], vocab, args.lower,
                                                args.max_post_length, args.max_sentence_length,
                                                frames=args.frames, discourse=args.discourse, words=args.words))
    
    rnn_params = dict(V=embeddings.shape[0],
                    d=embeddings.shape[1],
                    max_post_length=args.max_post_length,
                    max_sentence_length=args.max_sentence_length,
                    embeddings=embeddings,
                    num_layers=args.num_layers,
                    learning_rate=args.learning_rate,
                    op=args.op,
                    word_attn=args.word_attn,
                    sent_attn=args.sent_attn,
                    highway=args.highway,
                    hops=args.hops,
                    words=args.words,
                    frames=args.frames,
                    discourse=args.discourse)
            
    kwargs = dict(vocab=vocab,
                  rnn_params=rnn_params,
                  batch_size=args.batch_size,
                  num_epochs=args.num_epochs,
                  lambda_w=args.lambda_w,
                  word_dropout=args.word_dropout,
                  dropout=args.dropout,
                  early_stopping_heldout=args.early_stopping_heldout,
                  balance=args.balance,
                  pairwise=args.pairwise,
                  verbose=args.verbose)                  
    
    if biases:
        for split in inputs:
            inputs[split].append(np.array(biases[split]).reshape(len(biases[split]),1))
        kwargs['rnn_params'].update(add_biases=True)

    for split in inputs:
        inputs[split].append(data[split]['labels'])
    
    return inputs, kwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')
    parser.add_argument('outputfile')    

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--lambda_w')
    parser.add_argument('-n', '--num_layers', type=int, default=2)
    parser.add_argument('-l', '--learning_rate')
    parser.add_argument('--word_dropout')
    parser.add_argument('--dropout')
    
    parser.add_argument('--discourse', type=int, default=0)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--words', type=int, default=1)
    parser.add_argument('--biases', type=open)

    parser.add_argument('--max_post_length', type=int, default=40)
    parser.add_argument('--max_sentence_length', type=int, default=256)

    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--lower', action='store_true')
    
    parser.add_argument('--early_stopping_heldout', type=float, default=0)
    
    parser.add_argument('--balance', action='store_true')

    parser.add_argument('--verbose', action='store_true')        

    parser.add_argument('--op', action='store_true')
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--word_attn', type=int, default=1)
    parser.add_argument('--sent_attn', type=int, default=1)
    parser.add_argument('--highway', action='store_true')
    parser.add_argument('--pairwise', action='store_true')

    parser.add_argument('--vocab')

    parser.add_argument('--save_gold', action='store_true')
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--load', action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    #load training, testing, and embeddings
    print('loading data...')
    with open(args.inputfile) as f:
        data = json.load(f)

    if args.biases:
        biases = json.load(args.biases)
    else:
        biases = None

    if args.vocab and os.path.exists(args.vocab):
        with open(args.vocab) as f:
            vocab = json.load(f)
    else:
        print('building vocab...')
        vocab = build_vocab(data['train'], args.min_count, args.lower)
        if args.vocab:
            with open(args.vocab, 'w') as f:
                json.dump(vocab, f)

    inputs, kwargs = prepare(data, vocab, biases, args)
                       
    lambda_ws = [0, .0000001, .000001, .00001, .0001]
    if args.lambda_w:
        lambda_ws = map(float,args.lambda_w.split(',')) 
    
    learning_rates = [0.05, 0.01]
    if args.learning_rate:
        learning_rates = map(float,args.learning_rate.split(',')) 

    word_dropouts = [0.5, 0.25, 0, 0.75]
    if args.word_dropout:
        word_dropouts = map(float,args.word_dropout.split(',')) 
        
    dropouts = [0.25, 0, 0.5, 0.75]
    if args.dropout:
        dropouts = map(float,args.dropout.split(',')) 

    print('training...')
    if args.save_gold:
        np.save('{}.gold'.format(args.outputfile), val_y)

    training = inputs['train'][:-1]
    train_y = inputs['train'][-1]
    validation = inputs['val'][:-1]
    val_y = inputs['val'][-1]
        
    best = 0
            
    if args.load:
        classifier = load(args.outputfile, verbose=args.verbose)
        best = classifier.get_score(validation, val_y)
        if args.save_scores:
            scores = classifier.decision_function(validation)
            np.save('{}.scores'.format(args.outputfile), scores)

    for lambda_w in lambda_ws: 
        for learning_rate in learning_rates:
            for word_dropout in word_dropouts:
                for dropout in dropouts:
                    kwargs['rnn_params']['learning_rate'] = learning_rate
                    kwargs.update(dict(lambda_w=lambda_w,
                                       word_dropout=word_dropout,
                                       dropout=dropout))

                    classifier = PersuasiveInfluenceClassifier(**kwargs)
                    classifier.fit(training, train_y, validation, val_y)

                    score = classifier.get_score(validation, val_y)

                    if score > best:
                        best = score
                        classifier.save(args.outputfile)
                        if args.save_scores:
                            scores = classifier.decision_function(validation)
                            np.save('{}.scores'.format(args.outputfile), scores)
    
    if 'test' in data:
        score = classifier.get_score(inputs['test'][:-1], inputs['test'][-1])
