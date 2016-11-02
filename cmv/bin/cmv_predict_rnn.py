from __future__ import print_function

import argparse
import json

import numpy as np

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input

from cmv.rnn.preprocessing import build_indices
from cmv.rnn.argumentationRNN import ArgumentationRNN
from cmv.rnn.argumentationMetadataRNN import ArgumentationMetadataRNN
from cmv.rnn.argumentationEncoderDecoderRNN import ArgumentationEncoderDecoderRNN

def compile(data, decoder, no_shared, diff, sizes,
            dimension, recurrent_dimension,
            layers=1, pretrained=False):
    
    rnnType = ArgumentationRNN
    kwargs = {}
    if decoder:
        rnnType = ArgumentationEncoderDecoderRNN
        kwargs = dict(shared=not no_shared,
                      diff=diff)
    if len(sizes):
        rnnType = ArgumentationMetadataRNN
        kwargs = dict(pos='pos' in sizes,
                      deps='deps' in sizes,
                      govs='govs' in sizes,
                      frames=frames)
        argRNN = rnnType(sizes,
                        recurrent_dimension, data['train_rr_words'].shape[1],
                        data['train_rr_words'].shape[2], data['embeddings'], **kwargs)
    else:
        argRNN = rnnType(data['embeddings'].shape[0], data['embeddings'].shape[1],
                        recurrent_dimension, data['train_mask_rr_w'].shape[1],
                        data['train_mask_rr_w'].shape[2], data['embeddings'], **kwargs)

    return argRNN

def prepare(data, sizes, decoder):
    if len(sizes):
        training_inputs = [data['train_rr_words']]
        val_inputs = [data['val_rr_words']]
        if 'pos' in sizes:
            training_inputs += [data['train_rr_pos']]
            val_inputs += [data['val_rr_pos']]
        if 'deps' in sizes:
            training_inputs += [data['train_rr_deps']]
            val_inputs += [data['val_rr_deps']]
        if 'govs' in sizes:
            training_inputs += [data['train_rr_govs']]
            val_inputs += [data['val_rr_govs']]
        if 'frames' in sizes:
            training_inputs += [data['train_rr_frames']]
            val_inputs += [data['val_rr_frames']]
        training_inputs += [data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs += [data['val_mask_rr_w'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-2]
    elif decoder:
        training_inputs = [data['train_op'], data['train_rr'], 
                  data['train_mask_op_w'], data['train_mask_rr_w'], 
                  data['train_mask_op_s'], data['train_mask_rr_s']]
        val_inputs = [data['val_op'], data['val_rr'], 
                    data['val_mask_op_w'], data['val_mask_rr_w'], 
                    data['val_mask_op_s'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-4, len(training_inputs)-3]
    else:
        training_inputs = [data['train_rr'], data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs = [data['val_rr'], data['val_mask_rr_w'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-2]
    training_inputs += [data['train_labels']]
    val_inputs += [data['val_labels']]
    
    training = np.array(zip(*training_inputs))
    print(training.shape)
    validation = np.array(zip(*val_inputs))
    print(validation.shape)

    return training, validation, drop_indices

def train(data, kwargs, batch_size,
          num_restarts, num_epochs,
          lambda_w, word_dropout, dropout, 
          shuffle, outputfile):

    argRNN = compile(data, **kwargs['compile'])
    training, validation, drop_indices = prepare(data, **kwargs['prepare'])
    
    num_batches = training.shape[0] // batch_size
    if shuffle == 'nosep':
        num_batches = training.shape[0] // 2 // batch_size

    best_pairwise = 0
    best_accuracy = 0
    print(batch_size, num_batches, training.shape[0])
    for itr in range(num_restarts):
        for epoch in range(num_epochs):
            epoch_cost = 0
            #shuffle
            idxs = np.random.choice(training.shape[0], training.shape[0], False)
            if shuffle == 'nosep':
                idxs = np.random.choice(training.shape[0] // 2, training.shape[0] // 2, False)

            for batch_num in range(num_batches+1):
                s = batch_size * batch_num
                e = batch_size * (batch_num+1)
                if shuffle == 'nosep':
                    batch = np.concatenate([training[1::2][idxs[s:e]],
                                            training[::2][idxs[s:e]]], axis=0)
                elif shuffle == 'none':
                    batch = training[s:e]
                else:
                    batch = training[idxs[s:e]]

                inputs = zip(*batch)
                if word_dropout:
                    for drop_index in drop_indices:
                        inputs[drop_index] = inputs[drop_index]*(np.random.rand(*(np.array(inputs[drop_index]).shape)) < word_dropout)
                cost = argRNN.train(*(inputs+[lambda_w]))

                print(epoch, batch_num, cost)
                epoch_cost += cost
                if batch_num % (num_batches/2) == 0 and batch_num != 0:
                    scores = []
                    batch_size_v = validation.shape[0] // 10
                    for k in range(11):
                        s_v = batch_size_v * k
                        e_v = batch_size_v * (k+1)
                        scores += argRNN.predict(*(zip(*(validation[s_v:e_v]))[:-1])).tolist()

                    scores = np.array(scores)
                    ret = scores[::2] > scores[1::2]
                    print('Pairwise Accuracy: {}'.format(np.mean(ret)))
                    acc = ((scores > .5) == np.array(data['val_labels']))
                    print('Accuracy: {}'.format(np.mean(acc)))

                    if np.mean(ret) > best_pairwise:
                        best_pairwise = np.mean(ret)
                        argRNN.save(outputfile + '.pairwise.model')
                    if acc > best_accuracy:
                        best_accuracy = acc
                        argRNN.save(outputfile + '.accuracy.model')
            print('Total Epoch Cost: {}'.format(epoch_cost))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')
    parser.add_argument('outputfile')    
    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('--lambda_c', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-r', '--recurrent_dimension', type=int, default=100)
    parser.add_argument('--word_dropout', type=float, default=.25)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--no_shared', action='store_true')
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--shuffle', default='none')
    
    parser.add_argument('--sizes', type=json.loads, default={})
    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--deps', action='store_true')
    parser.add_argument('--govs', action='store_true')
    parser.add_argument('--frames', action='store_true')
    
    args = parser.parse_args()
    print(args)
    
    #load training, testing, and embeddings
    data = np.load(args.inputfile)
    
    rnnType = ArgumentationRNN
    kwargs = {}
    if args.decoder:
        rnnType = ArgumentationEncoderDecoderRNN
        kwargs = dict(shared=not args.no_shared,
                      diff=args.diff)
    if len(args.sizes):
        rnnType = ArgumentationMetadataRNN
        kwargs = dict(pos=args.pos,
                      deps=args.deps,
                      govs=args.govs,
                      frames=args.frames)
        argRNN = rnnType(args.sizes,
                        args.recurrent_dimension, data['train_rr_words'].shape[1],
                        data['train_rr_words'].shape[2], data['embeddings'], **kwargs)
    else:
        argRNN = rnnType(data['embeddings'].shape[0], data['embeddings'].shape[1],
                        args.recurrent_dimension, data['train_mask_rr_w'].shape[1],
                        data['train_mask_rr_w'].shape[2], data['embeddings'], **kwargs)

    print(kwargs)

    print(data['embeddings'].shape)
    #print(data['train_rr'].shape)
    print(data['train_mask_rr_s'].shape)
    print(data['train_mask_rr_w'].shape)
    print(data['train_labels'].shape)

    if len(args.sizes):
        training_inputs = [data['train_rr_words']]
        val_inputs = [data['val_rr_words']]
        if args.pos:
            training_inputs += [data['train_rr_pos']]
            val_inputs += [data['val_rr_pos']]
        if args.deps:
            training_inputs += [data['train_rr_deps']]
            val_inputs += [data['val_rr_deps']]
        if args.govs:
            training_inputs += [data['train_rr_govs']]
            val_inputs += [data['val_rr_govs']]
        if args.frames:
            training_inputs += [data['train_rr_frames']]
            val_inputs += [data['val_rr_frames']]
        training_inputs += [data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs += [data['val_mask_rr_w'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-2]
    elif args.decoder:
        training_inputs = [data['train_op'], data['train_rr'], 
                  data['train_mask_op_w'], data['train_mask_rr_w'], 
                  data['train_mask_op_s'], data['train_mask_rr_s']]
        val_inputs = [data['val_op'], data['val_rr'], 
                    data['val_mask_op_w'], data['val_mask_rr_w'], 
                    data['val_mask_op_s'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-4, len(training_inputs)-3]
    else:
        training_inputs = [data['train_rr'], data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs = [data['val_rr'], data['val_mask_rr_w'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-2]
    training_inputs += [data['train_labels']]
    val_inputs += [data['val_labels']]
    
    training = np.array(zip(*training_inputs))
    print(training.shape)
    validation = np.array(zip(*val_inputs))
    print(validation.shape)
        
    num_batches = training.shape[0] // args.batch_size
    if args.shuffle == 'nosep':
        num_batches = training.shape[0] // 2 // args.batch_size
    
    print(args.batch_size, num_batches, training.shape[0])
    best = 0    
    for epoch in range(args.num_epochs):
        epoch_cost = 0
        #shuffle
        #
        idxs = np.random.choice(training.shape[0], training.shape[0], False)
        if args.shuffle == 'nosep':
            idxs = np.random.choice(training.shape[0] // 2, training.shape[0] // 2, False)
            
        for batch_num in range(num_batches+1):
            s = args.batch_size * batch_num
            e = args.batch_size * (batch_num+1)
            if args.shuffle == 'nosep':
                batch = np.concatenate([training[1::2][idxs[s:e]],
                                        training[::2][idxs[s:e]]], axis=0)
            elif args.shuffle == 'none':
                batch = training[s:e]
            else:
                batch = training[idxs[s:e]]
            
            #train_op, train_rr, train_mask_op_w, train_mask_rr_w, train_mask_op_s, train_mask_rr_s, train_labels = zip(*batch)            
            #train_drop_mask_op_w = train_mask_op_w*(np.random.rand(*(np.array(train_mask_op_w).shape)) < args.word_dropout)
            #train_drop_mask_rr_w = train_mask_rr_w*(np.random.rand(*(np.array(train_mask_rr_w).shape)) < args.word_dropout)
            inputs = zip(*batch)
            if args.word_dropout:
                for drop_index in drop_indices:
                    inputs[drop_index] = inputs[drop_index]*(np.random.rand(*(np.array(inputs[drop_index]).shape)) < args.word_dropout)
            cost = argRNN.train(*(inputs+[args.lambda_w]))
            '''
            if args.decoder:
                cost = argRNN.train(train_op, train_rr,
                                    train_drop_mask_op_w, train_drop_mask_rr_w,
                                    train_mask_op_s, train_mask_rr_s,
                                    train_labels, args.lambda_w)
            else:
                cost = argRNN.train(train_rr,
                                    train_drop_mask_rr_w,
                                    train_mask_rr_s,
                                    train_labels, args.lambda_w)
            '''
            #cost = argRNN.train(*(zip(*training[s:e])+[args.lambda_w]))
            
            print(epoch, batch_num, cost)
            epoch_cost += cost
            if batch_num % (num_batches/2) == 0 and batch_num != 0:
                scores = []
                batch_size_v = validation.shape[0] // 10
                for k in range(11):
                    s_v = batch_size_v * k
                    e_v = batch_size_v * (k+1)
                    scores += argRNN.predict(*(zip(*(validation[s_v:e_v]))[:-1])).tolist()
                    '''
                    if args.decoder:
                        scores += argRNN.predict(data['val_op'][s_v:e_v],
                                                 data['val_rr'][s_v:e_v],
                                                 data['val_mask_op_w'][s_v:e_v],
                                                 data['val_mask_rr_w'][s_v:e_v],
                                                 data['val_mask_op_s'][s_v:e_v],
                                                 data['val_mask_rr_s'][s_v:e_v]).tolist()
                    else:
                        scores += argRNN.predict(data['val_rr'][s_v:e_v],
                                                 data['val_mask_rr_w'][s_v:e_v],
                                                 data['val_mask_rr_s'][s_v:e_v]).tolist()
                    '''
                scores = np.array(scores)
                ret = scores[::2] > scores[1::2]
                print('Pairwise Accuracy: {}'.format(np.mean(ret)))
                acc = ((scores > .5) == np.array(data['val_labels']))
                print('Accuracy: {}'.format(np.mean(acc)))

                if np.mean(ret) > best:
                    best = np.mean(ret)
                    argRNN.save(args.outputfile + '.model')
        print('Total Epoch Cost: {}'.format(epoch_cost))
