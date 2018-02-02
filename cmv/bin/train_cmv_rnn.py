from __future__ import print_function

import argparse
import json
import time
import collections

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cmv.rnn.persuasiveInfluenceClassifier import PersuasiveInfluenceClassifier
from cmv.rnn.vocab import build_vocab

def build_data(metadata, vocab, lower, 
               max_post_length, max_sentence_length, max_title_length,
               frames=False, discourse=False, words=True):
    print(max_post_length, max_sentence_length, max_title_length)
    
    data = collections.defaultdict(list)    
    new_metadata = collections.defaultdict(list)
    for split in ('train', 'val'):
        for name in ('pos', 'neg'):
            for index, post in enumerate(metadata[split+'_'+name]):
                if split+'_'+name+'_indices' in metadata:
                    op_index = metadata[split+'_'+name+'_indices'][index]
                    op = metadata[split+'_op'][op_index]
                    if split+'_titles' in metadata:
                        title = metadata[split+'_titles'][op_index]
                    else:
                        title = []
                else:
                    op = []
                    title = []
                new_metadata[split+'_op'].append(op)
                new_metadata[split+'_titles'].append(title)
                new_metadata[split+'_rr'].append(post)
                data[split+'_labels'].append(name=='pos')
                
    for split in ('train', 'val'):
        for name in ('op', 'rr', 'titles'):
            key = split+'_'+name
            idxs, mask, mask_s = prepare_data(new_metadata[key], vocab, lower,
                                              max_post_length, max_sentence_length,
                                              frames=frames, discourse=discourse, words=words)
            data[key] = idxs
            data[split+'_mask_'+name+'_w'] = mask
            data[split+'_mask_'+name+'_s'] = mask_s
            print(name, data[key].shape, data[split+'_mask_'+name+'_w'].shape, data[split+'_mask_'+name+'_s'].shape)

    data['embeddings'] = prepare_embeddings(metadata['embeddings'], vocab)
                
    return data

def prepare(data, vocab, frames, discourse, biases,
            max_post_length, max_sentence_length, max_title_length,
            min_count, lower, words=True, op=False):

    print('building data...')
    data = build_data(data, vocab, lower, 
                      max_post_length, max_sentence_length, max_title_length,
                      frames, discourse, words)

    kwargs = dict(V=data['embeddings'].shape[0],
                  d=data['embeddings'].shape[1],
                  max_post_length=max_post_length,
                  max_sentence_length=max_sentence_length,
                  max_title_length=max_title_length,
                  embeddings=data['embeddings'],
                  op=op)
    
    training_inputs = [data['train_rr'], data['train_mask_rr_w'], data['train_mask_rr_s']]
    val_inputs = [data['val_rr'], data['val_mask_rr_w'], data['val_mask_rr_s']]

    print(data['train_rr'].shape, data['train_mask_rr_w'].shape, data['train_mask_rr_s'].shape)
    print(data['val_rr'].shape, data['val_mask_rr_w'].shape, data['val_mask_rr_s'].shape)
    print(data['embeddings'].shape)
    t_sum = data['train_mask_rr_s'].sum(axis=1)
    print(t_sum.shape)
    print(np.argwhere(t_sum==0))

    v_sum = data['val_mask_rr_s'].sum(axis=1)
    print(v_sum.shape)
    print(np.argwhere(v_sum==0))
    
    if op: #CHANGE#
        training_inputs += [data['train_op'], data['train_mask_op_w'], data['train_mask_op_s']]
        val_inputs += [data['val_op'], data['val_mask_op_w'], data['val_mask_op_s']]

    #if title (TODO)
    #training_inputs += [data['train_titles'][:,0,:], data['train_mask_titles_w'][:,0,:]]
    #val_inputs += [data['val_titles'][:,0,:], data['val_mask_titles_w'][:,0,:]]
    
    if biases:
        training_inputs += [np.array(biases[0]).reshape(len(biases[0]),1)]
        val_inputs += [np.array(biases[1]).reshape(len(biases[1]),1)]
        kwargs.update(dict(add_biases=True))

    training_inputs.append(data['train_labels'])
    val_inputs.append(data['val_labels'])
        
    #training = np.array(zip(*training_inputs))
    #validation = np.array(zip(*val_inputs))
    training = training_inputs
    validation = val_inputs
    
    return training, data['train_labels'], validation, data['val_labels'], kwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')
    parser.add_argument('outputfile')    

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('-n', '--num_layers', type=int, default=0)
    parser.add_argument('-l', '--learning_rate', type=int, default=0)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    parser.add_argument('--discourse', default=0)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--words', type=int, default=1)
    parser.add_argument('--biases', type=open)

    parser.add_argument('--max_post_length', type=int, default=40)
    parser.add_argument('--max_sentence_length', type=int, default=256)
    parser.add_argument('--max_title_length', type=int, default=256)    

    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--lower', action='store_true')
    
    parser.add_argument('--early_stopping_heldout', type=float, default=0)
    
    parser.add_argument('--balance', action='store_true')

    parser.add_argument('--verbose', action='store_true')        

    parser.add_argument('--op', action='store_true')
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--pairwise', action='store_true')

    parser.add_argument('--vocab')
    
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

    training, y, validation, val_y, kwargs = prepare(data, vocab, args.frames, args.discourse, biases,
                                                     args.max_post_length, args.max_sentence_length,
                                                     args.max_title_length, args.min_count, args.lower, args.words, args.op)
    
    kwargs.update(dict(batch_size=args.batch_size,
                       num_epochs=args.num_epochs,
                       verbose=args.verbose,
                       early_stopping_heldout=args.early_stopping_heldout,
                       balance=args.balance,
                       hops=args.hops,
                       pairwise=args.pairwise))
                       
    #also tune:
    # learning rate (0.05, 0.01)
    # recurrent dimension (50, 100, 200, 300)
    # embedding dimension (50, 100, 200, 300)
    # layers (1,2)

    lambda_ws = [0, .0000001, .000001, .00001, .0001]
    if args.lambda_w:
        lambda_ws = [args.lambda_w] #TODO: args.lambda_w.split
    
    num_layerses = [2,1]
    if args.num_layers:
        num_layerses = [args.num_layers]
    
    learning_rates = [0.05, 0.01]
    if args.learning_rate:
        learning_rates = [args.learning_rate]

    word_dropouts = [0.5, 0.25, 0, 0.75]
    if args.word_dropout:
        word_dropouts = [args.word_dropout]
        
    dropouts = [0.25, 0, 0.5, 0.75]
    if args.dropout:
        dropouts = [args.dropout]

    print('training...')
    np.save('{}.gold'.format(args.outputfile), val_y)
    for lambda_w in lambda_ws: 
        for num_layers in num_layerses:
                for learning_rate in learning_rates:
                    for word_dropout in word_dropouts:
                        for dropout in dropouts:
                            outputfile = '{}.{}.{}.{}.{}.{}'.format(args.outputfile,
                                                                        lambda_w,
                                                                        word_dropout,
                                                                        dropout,
                                                                        num_layers,
                                                                        learning_rate)
                            kwargs.update(dict(lambda_w=lambda_w,
                                               num_layers=num_layers,
                                               learning_rate=learning_rate,
                                               word_dropout=word_dropout,
                                               dropout=dropout,
                                               outputfile=outputfile))
                            classifier = PersuasiveInfluenceClassifier(**kwargs)
                            classifier.fit(training, y, validation, val_y)
                            classifier.save(outputfile)
                            scores = classifier.decision_function(validation[:-1])
                            np.save('{}.scores'.format(outputfile), scores)
