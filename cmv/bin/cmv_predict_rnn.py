from __future__ import print_function

import argparse
import json
import time
import collections

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cmv.rnn.persuasiveInfluenceClassifier import PersuasiveInfluenceClassifier

def build_vocab(metadata, min_count, lower):
    print('min count is {}'.format(min_count))
    
    counts = collections.Counter()
    for name in ('op', 'pos', 'neg', 'titles'):
        if 'train_' + name not in metadata:
            print("ERROR: {} not in metadata".format(name))
            continue
        
        for post in metadata['train_' + name]:
            for sentence in post:
                for word in sentence['words']:
                    if lower:
                        word = word.lower()
                    counts[word] += 1

    vocab = {} #{'UNK': 0} #UNDO
    for name in ('op', 'pos', 'neg', 'titles'):
        print(name)
        if 'train_' + name not in metadata:
            print("ERROR: {} not in metadata".format(name))
            continue
        
        for post in metadata['train_' + name]:
            for index,sentence in enumerate(post):
                feature = 'INDEX_'+str(index)
                if feature not in vocab:
                    vocab[feature] = len(vocab)
                    
                for word in sentence['words']:
                    if lower:
                        word = word.lower()
                    if word not in vocab and counts[word] >= min_count:
                        vocab[word] = len(vocab)

                if 'frames' in sentence:
                    for frame in sentence['frames']:
                        if frame is None: #UNDO
                            continue
                        if frame not in vocab:
                            vocab[frame] = len(vocab)

                #UNDO
                if 'inter_discourse' in sentence:
                    discourse = sentence['inter_discourse']
                    if discourse is None:
                        continue
                    if discourse not in vocab:
                        vocab[discourse] = len(vocab)
                    
    #vocab['UNK'] = len(vocab)
    print('vocab size is {}'.format(len(vocab)))
    return vocab

def build_data(metadata, vocab, lower, 
               max_post_length, max_sentence_length, max_title_length,
               frames=False, discourse=False, sentiment=False, words=True):
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
            if name == 'titles':
                shape_w = (len(new_metadata[key]), max_post_length, max_title_length)
            else:
                shape_w = (len(new_metadata[key]), max_post_length, max_sentence_length)

            data[key] = np.zeros(shape_w)
            data[split+'_mask_'+name+'_w'] = np.zeros(shape_w)
            data[split+'_mask_'+name+'_s'] = np.zeros((len(new_metadata[key]), max_post_length))
            print(name, data[key].shape, data[split+'_mask_'+name+'_w'].shape, data[split+'_mask_'+name+'_s'].shape)
            
            for pindex,post in enumerate(new_metadata[key]):
                wctr = 0
                for sindex,sentence in enumerate(post):
                    if sindex >= max_post_length:
                        continue
                    
                    if words:
                        for windex,word in enumerate(sentence['words']):
                            if lower:
                                word = word.lower()
                            if word in vocab:
                                vindex = vocab[word]

                                if name == 'titles':
                                    if wctr < max_title_length:
                                        data[key][pindex][0][wctr] = vindex
                                        data[split+'_mask_'+name+'_w'][pindex][0][wctr] = 1
                                        wctr += 1
                                else:
                                    if windex < max_sentence_length:
                                        data[key][pindex][sindex][windex] = vindex
                                        data[split+'_mask_'+name+'_w'][pindex][sindex][windex] = 1
                            
                    #TODO: add frames and index
                    if frames and 'frames' in sentence:
                        for findex,frame in enumerate(sentence['frames']):
                            if frame is None or findex >= max_sentence_length:
                                continue
                            if frame not in vocab:
                                print(frame)
                                continue
                            vindex = vocab[frame]
                            data[key][pindex][sindex][findex] = vindex
                            data[split+'_mask_'+name+'_w'][pindex][sindex][findex] = 1

                    if discourse:
                        if 'inter_discourse' in sentence:
                            rel = sentence['inter_discourse']
                            vindex = vocab[rel]
                            data[key][pindex][sindex][0] = vindex

                        data[split+'_mask_'+name+'_s'][pindex][sindex] = 1
                        data[split+'_mask_'+name+'_w'][pindex][sindex][0] = 1

                    if data[split+'_mask_'+name+'_w'][pindex][sindex].sum() > 0:
                        data[split+'_mask_'+name+'_s'][pindex][sindex] = 1

                #make sure to add frames if there is no frame index between two
                if frames:
                    for sindex in range(max_post_length):
                        if data[split+'_mask_'+name+'_s'][pindex][sindex] and (sindex==0 or data[split+'_mask_'+name+'_s'][pindex][sindex-1]==1) and (sindex==max_post_length-1 or data[split+'_mask_'+name+'_s'][pindex][sindex+1]==1):
                            data[split+'_mask_'+name+'_s'][pindex][sindex] = 1
                            data[split+'_mask_'+name+'_w'][pindex][sindex][0] = 1
                            
                data[split+'_mask_'+name+'_w'][pindex][0][0] = 1
                data[split+'_mask_'+name+'_s'][pindex][0] = 1                
                        
    #finally, do embeddings
    dimension = len(metadata['embeddings'].values()[0])
    print('embeddings', len(vocab), dimension)
    embeddings_array = [None] * len(vocab)
    for word in vocab:
        if word in metadata['embeddings']:
            embeddings_array[vocab[word]] = metadata['embeddings'][word]
        else:
            embeddings_array[vocab[word]] = np.random.uniform(-1, 1, (dimension,))
   
    data['embeddings'] = np.array(embeddings_array)
    print(data['embeddings'].shape)
    
    return data

def prepare(data, vocab, frames, discourse, sentiment, biases,
            max_post_length, max_sentence_length, max_title_length,
            min_count, lower, words=True, op=False):

    print('building data...')
    data = build_data(data, vocab, lower, 
                      max_post_length, max_sentence_length, max_title_length,
                      frames, discourse, sentiment, words)
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
    parser.add_argument('--sentiment', type=int, default=0)
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

    print('building vocab...')
    vocab = build_vocab(data, args.min_count, args.lower)
        
    training, y, validation, val_y, kwargs = prepare(data, vocab, args.frames, args.discourse, args.sentiment, biases,
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

    'model_lower_cc_disc_wordsonly_redo.0.0.25.0.1.100.0.05'
    'model_lower_cc_disc_tagged_cc_intra.0.0.5.0.25.1.100.0.05'

    lambda_ws = [0, .0000001, .000001, .00001, .0001]
    if args.lambda_w:
        lambda_ws = [args.lambda_w]
    
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
