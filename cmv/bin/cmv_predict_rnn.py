from __future__ import print_function

import argparse
import json
import time
import collections

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input

from cmv.rnn.preprocessing import build_indices
from cmv.rnn.argumentationDAN import ArgumentationDAN
from cmv.rnn.argumentationRNN import ArgumentationRNN
from cmv.rnn.argumentationMetadataRNN import ArgumentationMetadataRNN
from cmv.rnn.argumentationEncoderDecoderRNN import ArgumentationEncoderDecoderRNN
from cmv.rnn.differenceNetwork import DifferenceNetwork
from cmv.rnn.argumentationBeliefState import ArgumentationBeliefState
from cmv.rnn.argumentationTransformation import ArgumentationTransformation
from cmv.rnn.argumentationDiscourseRNN import ArgumentationDiscourseRNN

def compile(data, rnn, decoder, belief, no_shared, diff, sizes, ignore_frames,
            dimension, recurrent_dimension, num_layers, learning_rate, chars,
            discourse_tagged, discourse_predictions, K, frames, intra_discourse, sentiment, causal,
            num_filters, filter_length_range,
            title, features, biases,
            pretrained=False):

    kwargs = dict(num_layers=num_layers, learning_rate=learning_rate)
    rnnType = ArgumentationRNN
    if decoder:
        rnnType = ArgumentationEncoderDecoderRNN
        if belief:
            rnnType = ArgumentationBeliefState

        kwargs = dict(shared=not no_shared,
                      diff=diff)
    elif diff:
        rnnType = DifferenceNetwork        
        if rnn:
            rnnType = ArgumentationTransformation
        
    elif not rnn:
        rnnType = ArgumentationDAN

    if len(sizes):
        rnnType = ArgumentationMetadataRNN
        kwargs = dict(frame_mask=ignore_frames is not None)
        return rnnType(sizes,
                        recurrent_dimension, data['train_rr_words'].shape[1],
                        data['train_rr_words'].shape[2], data['embeddings'], **kwargs)

    if 1: #discourse_tagged or discourse_predictions or frames:
        rnnType = ArgumentationDiscourseRNN
        if chars:
            V_chars = np.concatenate([data['train_rr_chars'],
                                      data['val_rr_chars']]).max()+1
            kwargs.update(dict(d_chars=chars,
                               V_chars=V_chars,
                               max_chars_length=data['train_rr_chars'].shape[-1],
                               filter_length_range=filter_length_range,
                               num_filters=num_filters))
            
        if discourse_tagged:
            K = np.concatenate([data['train_rr_inter_discourse'],
                                data['val_rr_inter_discourse']]).max()+1
            print(K)
            kwargs.update(dict(discourse_tagged=discourse_tagged,
                        discourse_predictions=discourse_predictions,
                        K=K))
        if frames:
            V_frames = np.concatenate([data['train_rr_frames'],
                                       data['val_rr_frames']]).max()+1
            kwargs.update(dict(d_frames=frames,
                               V_frames=V_frames))
        if intra_discourse:
            V_intra = np.concatenate([data['train_rr_intra_discourse'],
                                      data['val_rr_intra_discourse']]).max()+1
            kwargs.update(dict(d_intra=intra_discourse,
                               V_intra=V_intra))
        if sentiment:
            V_sentiment = np.concatenate([data['train_rr_sentiment'],
                                      data['val_rr_sentiment']]).max()+1
            kwargs.update(dict(d_sentiment=sentiment,
                               V_sentiment=V_sentiment))
        if causal:
            kwargs.update(causal=True)
            
    if title:
        kwargs.update(dict(d_title=title))

    if features:
        kwargs.update(dict(post_features_length=features[0].shape[1]))

    if biases:
        kwargs.update(dict(add_biases=True))
    print(kwargs)
        
    return rnnType(data['embeddings'].shape[0], data['embeddings'].shape[1],
                        recurrent_dimension, data['train_mask_rr_w'].shape[1],
                        data['train_mask_rr_w'].shape[2], data['embeddings'], **kwargs)

def get_frame_mask(frames, mask, ignore_frames):
    print('getting mask for ', ignore_frames)
    frame_mask = np.array(mask)
    ignore_frames = {int(i) for i in ignore_frames.split(',')}
    for post in range(frame_mask.shape[0]):
        for sentence in range(frame_mask.shape[1]):
            for word in range(frame_mask.shape[2]):
                if frames[post][sentence][word] in ignore_frames:
                    frame_mask[post][sentence][word] = 0

    print(frame_mask[0])
    return frame_mask

def prepare(data, sizes, decoder, diff, ignore_frames, chars, discourse_tagged, discourse_predictions,
            lambda_k, frames, intra_discourse, sentiment, causal, title, features, biases):
    train_op = 'train_op'
    train_rr = 'train_rr'
    val_op = 'val_op'
    val_rr = 'val_rr'
    if train_op not in data:
        train_op = 'train_op_words'
        train_rr = 'train_rr_words'
        val_op = 'val_op_words'
        val_rr = 'val_rr_words'
        
    if len(sizes):
        print('in sizes')
        training_inputs = [data[train_rr]]
        val_inputs = [data[val_rr]]
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
        if 'clusters' in sizes:
            training_inputs += [data['train_rr_clusters']]
            val_inputs += [data['val_rr_clusters']]
        training_inputs += [data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs += [data['val_mask_rr_w'], data['val_mask_rr_s']]
        if ignore_frames is not None:
            training_inputs.append(get_frame_mask(data['train_rr_frames'],
                                                  data['train_mask_rr_w'],
                                                  ignore_frames))
            val_inputs.append(get_frame_mask(data['val_rr_frames'],
                                             data['val_mask_rr_w'],
                                             ignore_frames))

        drop_indices = [len(training_inputs)-2]
    elif decoder or diff:
        training_inputs = [data[train_op], data[train_rr], 
                  data['train_mask_op_w'], data['train_mask_rr_w'], 
                  data['train_mask_op_s'], data['train_mask_rr_s']]
        val_inputs = [data[val_op], data[val_rr], 
                    data['val_mask_op_w'], data['val_mask_rr_w'], 
                    data['val_mask_op_s'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-4, len(training_inputs)-3]
    else:
        training_inputs = [data[train_rr], data['train_mask_rr_w'], data['train_mask_rr_s']]
        val_inputs = [data[val_rr], data['val_mask_rr_w'], data['val_mask_rr_s']]
        drop_indices = [len(training_inputs)-2]
    if discourse_tagged or frames or chars:
        if chars:
            training_inputs.append(data['train_rr_chars'])
            val_inputs.append(data['val_rr_chars'])
            
        if discourse_tagged:
            training_inputs.append(data['train_rr_inter_discourse'])
            val_inputs.append(data['val_rr_inter_discourse'])
                    
        if discourse_predictions:
            training_inputs.append(lambda_k)
            val_inputs.append(lambda_k)

        if frames:
            training_inputs.append(data['train_rr_frames'])
            val_inputs.append(data['val_rr_frames'])
        if intra_discourse:
            training_inputs.append(data['train_rr_intra_discourse'].max(axis=-1))
            val_inputs.append(data['val_rr_intra_discourse'].max(axis=-1))
        if sentiment:
            training_inputs.append(data['train_rr_sentiment'])
            val_inputs.append(data['val_rr_sentiment'])
        if causal:
            training_inputs.append(data['train_rr_causality'])#.max(axis=-1).reshape((data['train_rr_causality'].shape[0],-1,1)))
            val_inputs.append(data['val_rr_causality'])#.max(axis=-1).reshape((data['val_rr_causality'].shape[0],-1,1)))
        
    if title:
        training_inputs.extend([data['train_titles'], data['train_mask_titles']])
        val_inputs.extend([data['val_titles'], data['val_mask_titles']])

    if features:
        training_inputs += [features[0]]
        val_inputs += [features[1]]

    if biases:
        training_inputs += [np.array(biases[0]).reshape(len(biases[0]),1)]
        val_inputs += [np.array(biases[1]).reshape(len(biases[1]),1)]
                        
    training_inputs += [data['train_labels']]
    val_inputs += [data['val_labels']]
    
    training = np.array(zip(*training_inputs))
    print(training.shape)
    validation = np.array(zip(*val_inputs))
    print(validation.shape)

    return training, validation, drop_indices

def train(data, kwargs, batch_size,
          num_restarts, num_epochs,
          lambda_w, word_dropout, dropout, belief, discourse_predictions,
          shuffle, outputfile, belief_weight=1.0,
          pairwise=False, accuracy=False, f_score=False, auc=False, balance=False):

    print(outputfile)

    training, validation, drop_indices = prepare(data, **kwargs['prepare'])
    
    num_batches = training.shape[0] // batch_size
    if shuffle == 'nosep':
        num_batches = training.shape[0] // 2 // batch_size

    best_pairwise = 0
    best_accuracy = 0
    best_f_score = 0
    best_auc_score = 0
    print(batch_size, num_batches, training.shape[0])
    for itr in range(num_restarts):
        print('restart: {}'.format(itr))
        argRNN = compile(data, **kwargs['compile'])
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            if discourse_predictions:
                print(True)
                epoch_counts = np.zeros(kwargs['compile']['K'])
            start_time = time.time()
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
                if belief:
                    cost = argRNN.train(*(inputs+[lambda_w, dropout, belief_weight]))
                elif discourse_predictions:
                    cost, counts = argRNN.train(*(inputs+[lambda_w, dropout]))
                    epoch_counts += counts.sum(axis=0)
                    print(counts.sum(axis=0))
                else:
                    weights = np.ones_like(inputs[-1])
                    if balance:
                        label_counts = collections.Counter(inputs[-1])
                        max_count = 1.*max(label_counts.values())
                        class_weights = {i:1/(label_counts[i]/max_count) for i in label_counts}
                        print(label_counts, class_weights)
                        weights = np.array([class_weights[i] for i in inputs[-1]]).astype(np.float32)
                        
                    cost = argRNN.train(*(inputs+[lambda_w, dropout, weights]))

                print(epoch, batch_num, cost)
                epoch_cost += cost
                
            #if batch_num % (num_batches/2) == 0 and batch_num != 0:
            scores = []
            if discourse_predictions:
                counts = np.zeros(kwargs['compile']['K'])
            batch_size_v = validation.shape[0] // 10
            for k in range(11):
                s_v = batch_size_v * k
                e_v = batch_size_v * (k+1)
                if discourse_predictions:
                    batch_scores, batch_counts = argRNN.predict(*(zip(*(validation[s_v:e_v]))[:-1]))
                    scores += batch_scores.tolist()
                    counts += batch_counts.sum(axis=0)                            
                else:
                    scores += argRNN.predict(*(zip(*(validation[s_v:e_v]))[:-1])).tolist()

            scores = np.array(scores)

            if pairwise:
                ret = scores[::2] > scores[1::2]
                print('{} Pairwise Accuracy: {}'.format(outputfile, np.mean(ret)))
                print(np.sum(scores <= .5), np.sum(scores > .5))

                if np.mean(ret) > best_pairwise:
                    best_pairwise = np.mean(ret)
                    argRNN.save(outputfile + '.pairwise.model')
                    np.save(outputfile + '.pairwise.predictions', ret)

            if accuracy:
                acc = ((scores > .5) == np.array(data['val_labels']))
                print('{} Accuracy: {}'.format(outputfile, np.mean(acc)))

                if np.mean(acc) > best_accuracy:
                    best_accuracy = np.mean(acc)
                    argRNN.save(outputfile + '.accuracy.model')
                    np.save(outputfile + '.predictions', scores > .5)

            if f_score:
                predictions = scores > .5
                precision, recall, fscore, _ = precision_recall_fscore_support(data['val_labels'],
                                                                               predictions)
                print('{} Precision: {} Recall: {} Fscore: {}'.format(outputfile,
                                                                      precision,
                                                                      recall,
                                                                      fscore))
                if fscore[1] > best_f_score:
                    best_f_score = fscore[1]
                    argRNN.save(outputfile + 'fscore.model')
                    np.save(outputfile + '.predictions', predictions)
            if auc:
                auc_score = roc_auc_score(data['val_labels'], scores)
                print('{} ROC AUC: {}'.format(outputfile, auc_score))
                if auc_score > best_auc_score:
                    best_auc_score = auc_score
                    argRNN.save(outputfile + 'auc_score.model')
                    np.save(outputfile + '.scores', scores)
                    
            epoch_time = time.time() - start_time
            print('Total Epoch Cost for {}: {}'.format(outputfile, epoch_cost))
            print('Total Epoch Time: {}'.format(epoch_time))
            if discourse_predictions:
                print('Total Epoch Counts for {}: {}'.format(outputfile, epoch_counts))
                print('Total Validation Counts for {}: {}'.format(outputfile, counts)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')
    parser.add_argument('outputfile')    
    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('--lambda_k', type=float, default=0)

    parser.add_argument('--num_restarts', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('-r', '--recurrent_dimension', type=int, default=100)
    parser.add_argument('-n', '--num_layers', type=int, default=1)
    parser.add_argument('-l', '--learning_rate', type=int, default=0.01)
    parser.add_argument('--word_dropout', type=float, default=.25)
    parser.add_argument('--dropout', type=float, default=.25)
    
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--belief', action='store_true')
    parser.add_argument('--belief_weight', type=float, default=1.0)
    parser.add_argument('--rnn', action='store_true')
    parser.add_argument('--chars', type=int, default=0)
    parser.add_argument('--discourse_tagged', action='store_true')
    parser.add_argument('--discourse_predictions', action='store_true')    
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--intra_discourse', type=int, default=0)
    parser.add_argument('--sentiment', type=int, default=0)
    parser.add_argument('--causal', action='store_true')

    parser.add_argument('--title', type=int, default=0)

    parser.add_argument('--features', type=open)
    parser.add_argument('--filter')
    parser.add_argument('--biases', type=open)
    
    parser.add_argument('--no_shared', action='store_true')
    parser.add_argument('--diff', action='store_true')
    parser.add_argument('--shuffle', default='none', choices=('none','nosep','random'))
    
    parser.add_argument('--sizes', type=json.loads, default={})
    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--deps', action='store_true')
    parser.add_argument('--govs', action='store_true')
    parser.add_argument('--ignore_frames')

    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--f_score', action='store_true')
    parser.add_argument('--auc', action='store_true')

    parser.add_argument('--balance', action='store_true')        
            
    args = parser.parse_args()
    print(args)
    
    #load training, testing, and embeddings
    data = np.load(args.inputfile)

    if args.features:
        features = json.load(args.features)
        train_features = pd.read_json(features['train_features']).sort_index()
        val_features = pd.read_json(features['val_features']).sort_index()
        if args.filter:
            feature_names = [i for i in train_features.columns if args.filter in i]
        else:
            feature_names = [i for i in train_features.columns if i not in ('index','label')]
        features = [np.array(train_features[feature_names]),
                    np.array(val_features[feature_names])]
    else:
        features = None

    if args.biases:
        biases = json.load(args.biases)
    else:
        biases = None
        
    if args.decoder:
        missing = np.argwhere(data['train_mask_op_w'].sum(axis=(1,2))== 0).ravel()
        idxs = sorted(set(np.arange(data['train_mask_op_w'].shape[0])) - set(missing))
        new_data = {}
        for key in data:
            if 'train' in key:
                new_data[key] = data[key][idxs]
            else:
                new_data[key] = data[key]
        data = new_data
    
    print(data['embeddings'].shape)
    #print(data['train_rr'].shape)
    print(data['train_mask_rr_s'].shape)
    print(data['train_mask_rr_w'].shape)
    print(data['train_labels'].shape)

    kwargs = dict(compile=dict(decoder=args.decoder,
                               belief=args.belief,
                               rnn=args.rnn,
                               no_shared=args.no_shared,
                               diff=args.diff,
                               sizes=args.sizes,
                               dimension=200,
                               recurrent_dimension=args.recurrent_dimension,
                               learning_rate=args.learning_rate,
                               num_layers=args.num_layers,
                               ignore_frames=args.ignore_frames,
                               chars=args.chars,
                               discourse_tagged=args.discourse_tagged,
                               discourse_predictions=args.discourse_predictions,
                               K=args.K,
                               frames=args.frames,
                               intra_discourse=args.intra_discourse,
                               sentiment=args.sentiment,
                               causal=args.causal,
                               title=args.title,
                               features=features,
                               biases=biases),
                  prepare=dict(sizes=args.sizes,
                               decoder=args.decoder,
                               diff=args.diff,
                               discourse_tagged=args.discourse_tagged,
                               discourse_predictions=args.discourse_predictions,
                               lambda_k=args.lambda_k,
                               ignore_frames=args.ignore_frames,
                               chars=args.chars,
                               frames=args.frames,
                               intra_discourse=args.intra_discourse,
                               sentiment=args.sentiment,
                               causal=args.causal,
                               title=args.title,
                               features=features,
                               biases=biases))

    word_dropout=args.word_dropout
    dropout=args.dropout

    #also tune:
    # learning rate (0.05, 0.01)
    # recurrent dimension (50, 100, 200, 300)
    # embedding dimension (50, 100, 200, 300)
    # layers (1,2)

    'model_lower_cc_disc_wordsonly_redo.0.0.25.0.1.100.0.05'
    'model_lower_cc_disc_tagged_cc_intra.0.0.5.0.25.1.100.0.05'

    lambda_ws = [0] #[0, .0000001, .000001, .00001, .0001]
    num_layerses = [2] #[2,1]
    recurrent_dimensions = [100, 50, 200, 300]
    learning_rates = [0.05, 0.01]
    word_dropouts = [0.5, 0.25, 0, 0.75]
    dropouts = [0.25, 0, 0.5, 0.75]
    num_filterses = ['NA'] #[20, 30, 50, 100]
    filter_length_ranges = ['NA'] #[(1,1), (1,2), (1,3), (1,4), (1,5)]
    
    for lambda_w in lambda_ws: 
        for num_layers in num_layerses:
            kwargs['compile']['num_layers'] = num_layers            
            for recurrent_dimension in recurrent_dimensions:
                kwargs['compile']['recurrent_dimension'] = recurrent_dimension
                for learning_rate in learning_rates:
                    kwargs['compile']['learning_rate'] = learning_rate
                    for word_dropout in word_dropouts:
                        for dropout in dropouts:
                            for num_filters in num_filterses:
                                kwargs['compile']['num_filters'] = num_filters
                                for filter_length_range in filter_length_ranges:
                                    kwargs['compile']['filter_length_range'] = filter_length_range

                                    train(data, kwargs, args.batch_size,
                                          args.num_restarts, args.num_epochs,
                                          lambda_w, word_dropout, dropout, args.belief,
                                          args.discourse_predictions,
                                          args.shuffle,
                                          '{}.{}.{}.{}.{}.{}.{}.{}.{}'.format(args.outputfile,
                                                                        lambda_w,
                                                                        word_dropout,
                                                                        dropout,
                                                                        num_layers,
                                                                        recurrent_dimension,
                                                                        learning_rate,
                                                                        num_filters,
                                                                        filter_length_range),
                                            args.belief_weight,
                                            accuracy=args.accuracy,
                                            pairwise=args.pairwise,
                                            f_score=args.f_score,
                                            auc=args.auc,
                                            balance=args.balance)

