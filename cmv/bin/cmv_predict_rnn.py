from __future__ import print_function

import argparse

import numpy as np

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input

from cmv.rnn.preprocessing import build_indices
from cmv.rnn.argumentationRNN import ArgumentationRNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')
    parser.add_argument('outputfile')    
    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('--lambda_c', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('-r', '--recurrent_dimension', type=int, default=100)
    parser.add_argument('--word_dropout', type=float, default=.25)
        
    args = parser.parse_args()

    #load training, testing, and embeddings
    data = np.load(args.inputfile)
    
    argRNN = ArgumentationRNN(data['embeddings'].shape[0], data['embeddings'].shape[1],
                              args.recurrent_dimension, data['train_op'].shape[1],
                              data['train_op'].shape[2], data['embeddings'])

    print(data['embeddings'].shape)
    print(data['train_rr'].shape)
    print(data['train_mask_rr_s'].shape)
    print(data['train_mask_rr_w'].shape)
    print(data['train_labels'].shape)
    
    batch_size = data['train_op'].shape[0] // args.num_batches
    best = 0
    for epoch in range(args.num_epochs):
        training = zip(data['train_op'],
                       data['train_rr'],
                       data['train_mask_op_w'],
                       data['train_mask_rr_w'],
                       data['train_mask_op_s'],
                       data['train_mask_rr_s'],
                       data['train_labels'])
        for batch in range(args.num_batches+1):
            s = batch_size * batch
            e = batch_size * (batch+1)
            train_op, train_rr, train_mask_op_w, train_mask_rr_w, train_mask_op_s, train_mask_rr_s, train_labels = zip(*training[s:e])
            train_drop_mask_op_w = train_mask_op_w*(np.random.rand(*(np.array(train_mask_op_w).shape)) < args.word_dropout)
            train_drop_mask_rr_w = train_mask_rr_w*(np.random.rand(*(np.array(train_mask_rr_w).shape)) < args.word_dropout)

            cost = argRNN.train(train_op, train_rr,
                                train_drop_mask_op_w, train_drop_mask_rr_w,
                                train_mask_op_s, train_mask_rr_s,
                                train_labels, args.lambda_w)
            #cost = argRNN.train(*(zip(*training[s:e])+[args.lambda_w]))
            
            print(epoch, batch, cost)
            if batch % (args.num_batches/2) == 0 and batch != 0:
                scores = []
                batch_size_v = data['val_op'].shape[0] // 10
                for k in range(11):
                    s_v = batch_size_v * k
                    e_v = batch_size_v * (k+1)
                    scores += argRNN.predict(data['val_op'][s_v:e_v],
                                             data['val_rr'][s_v:e_v],
                                             data['val_mask_op_w'][s_v:e_v],
                                             data['val_mask_rr_w'][s_v:e_v],
                                             data['val_mask_op_s'][s_v:e_v],
                                             data['val_mask_rr_s'][s_v:e_v]).tolist()
                scores = np.array(scores)
                ret = scores[::2] > scores[1::2]
                print('Pairwise Accuracy: {}'.format(np.mean(ret)))
                acc = ((scores > .5) == np.array(data['val_labels']))
                print('Accuracy: {}'.format(np.mean(acc)))

                if np.mean(ret) > best:
                    best = np.mean(ret)
                    argRNN.save(args.outputfile + '.model')
