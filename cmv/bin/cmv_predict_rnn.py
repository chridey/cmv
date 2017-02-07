from __future__ import print_function

import argparse
import json
import time
import collections

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cmv.rnn.persuasiveInfluenceClassifier import PersuasiveInfluenceClassifier
    
def prepare(data, frames, discourse, sentiment, biases):
    kwargs = dict(V=data['embeddings'].shape[0],
                  d=data['embeddings'].shape[1],
                  max_post_length=data['train_mask_rr_w'].shape[1],
                  max_sentence_length=data['train_mask_rr_w'].shape[2],
                  embeddings=data['embeddings'])

    training_inputs = [data['train_rr_words'], data['train_mask_rr_w'], data['train_mask_rr_s']]
    val_inputs = [data['val_rr_words'], data['val_mask_rr_w'], data['val_mask_rr_s']]

    if frames:
        V_frames = np.concatenate([data['train_rr_frames'],
                                data['val_rr_frames']]).max()+1
        kwargs.update(dict(d_frames=frames,
                    V_frames=V_frames))
        training_inputs.append(data['train_rr_frames'])
        val_inputs.append(data['val_rr_frames'])
        
    if discourse:
        training_inputs.append(data['train_rr_inter_discourse'])
        val_inputs.append(data['val_rr_inter_discourse'])
                    
        V_inter = np.concatenate([data['train_rr_inter_discourse'],
                            data['val_rr_inter_discourse']]).max()+1
        V_intra = np.concatenate([data['train_rr_intra_discourse'],
                                  data['val_rr_intra_discourse']]).max()+1
        kwargs.update(dict(V_inter=V_inter,
                           d_intra=discourse,
                           V_intra=V_intra))

    if sentiment:
        training_inputs.append(data['train_rr_sentiment'])
        val_inputs.append(data['val_rr_sentiment'])
        V_sentiment = np.concatenate([data['train_rr_sentiment'],
                                    data['val_rr_sentiment']]).max()+1
        kwargs.update(dict(d_sentiment=sentiment,
                    V_sentiment=V_sentiment))

    if biases:
        training_inputs += [np.array(biases[0]).reshape(len(biases[0]),1)]
        val_inputs += [np.array(biases[1]).reshape(len(biases[1]),1)]
        kwargs.update(dict(add_biases=True))

    training_inputs.append(data['train_labels'])
    val_inputs.append(data['val_labels'])
        
    training = np.array(zip(*training_inputs))
    validation = np.array(zip(*val_inputs))

    return training, data['train_labels'], validation, data['val_labels'], kwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('inputfile')

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('-r', '--recurrent_dimension', type=int, default=0)
    parser.add_argument('-n', '--num_layers', type=int, default=0)
    parser.add_argument('-l', '--learning_rate', type=int, default=0)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    parser.add_argument('--discourse', default=0)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--sentiment', type=int, default=0)
    parser.add_argument('--biases', type=open)

    parser.add_argument('--early_stopping_heldout', type=float, default=0)
    
    parser.add_argument('--balance', action='store_true')

    parser.add_argument('--verbose', action='store_true')        
            
    args = parser.parse_args()
    print(args)
    
    #load training, testing, and embeddings
    data = np.load(args.inputfile)

    if args.biases:
        biases = json.load(args.biases)
    else:
        biases = None
        
    print(data['embeddings'].shape)
    print(data['train_mask_rr_s'].shape)
    print(data['train_mask_rr_w'].shape)
    print(data['train_labels'].shape)

    training, y, validation, val_y, kwargs = prepare(data, args.frames, args.discourse, args.sentiment, biases)
    kwargs.update(dict(batch_size=args.batch_size,
                       num_epochs=args.num_epochs,
                       verbose=args.verbose,
                       early_stopping_heldout=args.early_stopping_heldout,
                       balance=args.balance))
                       
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
    
    recurrent_dimensions = [100, 50, 200, 300]
    if args.recurrent_dimension:
        recurrent_dimensions = [args.recurrent_dimension]

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
        
    for lambda_w in lambda_ws: 
        for num_layers in num_layerses:
            for recurrent_dimension in recurrent_dimensions:
                for learning_rate in learning_rates:
                    for word_dropout in word_dropouts:
                        for dropout in dropouts:
                            kwargs.update(dict(lambda_w=lambda_w,
                                               num_layers=num_layers,
                                               rd=recurrent_dimension,
                                               learning_rate=learning_rate,
                                               word_dropout=word_dropout,
                                               dropout=dropout))
                            classifier = PersuasiveInfluenceClassifier(**kwargs)
                            classifier.fit(training, y)
                            classifier.save('{}.{}.{}.{}.{}.{}.{}.{}.{}'.format(args.outputfile,
                                                                        lambda_w,
                                                                        word_dropout,
                                                                        dropout,
                                                                        num_layers,
                                                                        recurrent_dimension,
                                                                        learning_rate))
                            scores = classifier.decision_function(validation, val_y)
                            np.save('{}.{}.{}.{}.{}.{}.{}.{}.{}.scores'.format(args.outputfile,
                                                                        lambda_w,
                                                                        word_dropout,
                                                                        dropout,
                                                                        num_layers,
                                                                        recurrent_dimension,
                                                                        learning_rate), scores)
