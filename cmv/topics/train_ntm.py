import time
import argparse
import json
import cPickle
import collections

import lasagne
import gensim
import theano
import theano.tensor as T
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from cmv.topics import utils
from cmv.topics.layers import MyEmbeddingLayer, AverageLayer, AverageNegativeLayer, ReconLayer
from cmv.rnn.layers import AttentionWordLayer, TopicAttentionWordLayer, WeightedAverageWordLayer, AttentionSentenceLayer, WeightedAverageSentenceLayer

'''
    lambda_ws = [0] #[0, .0000001, .000001, .00001, .0001]                                                                                                                                                                                    
    num_layerses = [2] #[2,1]                                                                                                                                                                                                                 
    recurrent_dimensions = [100, 50, 200, 300]
    learning_rates = [0.05, 0.01]
    word_dropouts = [0.5, 0.25, 0, 0.75]
    dropouts = [0.25, 0, 0.5, 0.75]
    num_filterses = ['NA'] #[20, 30, 50, 100]                                                                                                                                                                                                 
    filter_length_ranges = ['NA'] #[(1,1), (1,2), (1,3), (1,4), (1,5)] 
'''

# assemble the network
def build_ntm(d_word, len_voc, 
    num_descs, max_len, We,
    max_post_length, max_sentence_length,
    len_voc_rr, We_rr, rd=100,
    freeze_words=True, eps=1e-5, lr=0.01, negs=10,
    num_layers=2, add_biases=False,
              GRAD_CLIP=100, topic=False, influence=False, lambda_t=1.0, combined=False, sentence_attention=False):

    # input theano vars
    in_words = T.imatrix(name='words')
    in_neg = T.itensor3(name='neg')
    in_currmasks = T.matrix(name='curr_masks')
    in_dropmasks = T.matrix(name='drop_masks')
    in_negmasks = T.tensor3(name='neg_masks')

    # define network
    l_inwords = lasagne.layers.InputLayer(shape=(None, max_len), 
        input_var=in_words)
    l_inneg = lasagne.layers.InputLayer(shape=(None, negs, max_len), 
        input_var=in_neg)
    l_currmask = lasagne.layers.InputLayer(shape=(None, max_len), 
        input_var=in_currmasks)
    l_dropmask = lasagne.layers.InputLayer(shape=(None, max_len), 
        input_var=in_dropmasks)
    l_negmask = lasagne.layers.InputLayer(shape=(None, negs, max_len), 
        input_var=in_negmasks)

    #embeddings are now B x L x D
    l_emb = MyEmbeddingLayer(l_inwords, len_voc, 
        d_word, W=lasagne.utils.floatX(We), name='word_emb')
    # negative examples should use same embedding matrix
    # B x N x L x D
    l_negemb = MyEmbeddingLayer(l_inneg, len_voc, 
            d_word, W= l_emb.W, name='word_emb_copy1')

    # freeze embeddings
    if freeze_words:
        l_emb.params[l_emb.W].remove('trainable')
        l_negemb.params[l_negemb.W].remove('trainable')

    # average each post's embeddings
    # B x D
    l_currsum = AverageLayer([l_emb, l_currmask], d_word)
    l_dropsum = AverageLayer([l_emb, l_dropmask], d_word)
    # B x N x D
    l_negsum = AverageNegativeLayer([l_negemb, l_negmask], d_word)

    # pass all embeddings thru feed-forward layer
    l_mix = lasagne.layers.DenseLayer(l_dropsum, d_word)

    # compute weights over dictionary
    # B x K
    l_rels = lasagne.layers.DenseLayer(l_mix, num_descs,
                                       nonlinearity=lasagne.nonlinearities.softmax)

    # multiply weights with dictionary matrix
    # now B x D again
    #l_recon = lasagne.layers.DenseLayer(l_rels, d_word, b=None, nonlinearity=None)
    l_recon = ReconLayer(l_rels, d_word, num_descs)
    
    # compute loss
    currsums = lasagne.layers.get_output(l_currsum)
    negsums = lasagne.layers.get_output(l_negsum)
    recon = lasagne.layers.get_output(l_recon)

    currsums /= currsums.norm(2, axis=1)[:, None]
    recon /= recon.norm(2, axis=1)[:, None]
    negsums /= negsums.norm(2, axis=-1)[:, :, None]
    #now B
    correct = T.sum(recon * currsums, axis=1)
    #now B x N
    negs = T.sum(recon[:, None, :] * negsums, axis=-1) 

    #tloss = T.sum(T.maximum(0., 1. - correct[:, None] + negs))
    #add normalized sum, so that loss is of same magnitude
    #tloss = T.mean(T.maximum(0., 1. - correct[:, None] + negs))
    topic_weights = T.vector('topic_weights')
    tloss = T.sum(T.maximum(0., 1. - correct[:, None] + negs), axis=-1)
    weighted_tloss = lambda_t * lasagne.objectives.aggregate(tloss, topic_weights, mode='normalized_sum')
    tloss = T.sum(tloss)
        
    # enforce orthogonality constraint
    norm_R = l_recon.R / l_recon.R.norm(2, axis=1)[:, None]
    ortho_penalty = eps * T.sum((T.dot(norm_R, norm_R.T) - \
        T.eye(norm_R.shape[0])) ** 2)
    ntm_loss = tloss + ortho_penalty

    #now combine the topic modeling with the persuasive influence prediction
    #B x S x W tensor of batches of posts
    idxs_rr = T.itensor3('idxs_rr')
    mask_rr_w = T.itensor3('mask_rr_w')
    #B x S matrix
    mask_rr_s = T.imatrix('mask_rr_s')
    
    #B-long vector
    gold = T.ivector('gold')
    biases = T.matrix('biases')
    weights = T.vector('weights')

    #scalar parameters
    lambda_w = T.scalar('lambda_w')
    p_dropout = T.scalar('p_dropout')
    
    #now use this as an input to an LSTM
    l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                          input_var=idxs_rr)
    l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=mask_rr_w)
    l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                            input_var=mask_rr_s)
    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                         input_var=biases)
    
    #now B x S x W x D
    l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, len_voc_rr, d_word,
                                               W=lasagne.utils.floatX(We_rr))
    #CBOW w/attn
    #now B x S x D

    l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], rd)
    attention_layers = [l_attn_rr_w, None]    
    if topic and influence:
        l_attn_rr_w = TopicAttentionWordLayer([l_emb_rr_w, l_mask_rr_w, l_rels], rd)
        attention_layers[1] = l_attn_rr_w
        if combined:
            l_attn_rr_w = lasagne.layers.ConcatLayer(attention_layers, axis=-1)
        else:
            attention_layers[0] = None

    l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

    if sentence_attn:
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)

        #LSTM w/ attn
        #now B x D
        #TODO: is this attention layer needed? just take avg or last state of forward/backward?
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        l_hid = l_lstm_rr_avg
    else:
        l_lstm_rr_s_fwd = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        l_lstm_rr_s_rev = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                                   mask_input=l_mask_rr_s,
                                                   backwards=True)

        l_lstm_rr_s_fwd_slice = lasagne.layers.SliceLayer(l_lstm_rr_s_fwd, indices=-1, axis=1)
        l_lstm_rr_s_rev_slice = lasagne.layers.SliceLayer(l_lstm_rr_s_rev, indices=-1, axis=1)
        l_lstm_rr_s_bi = lasagne.layers.ConcatLayer([l_lstm_rr_s_fwd_slice, l_lstm_rr_s_rev_slice], axis=-1)
        l_hid = l_lstm_rr_s_bi

    for num_layer in range(num_layers):
        l_hid = lasagne.layers.DenseLayer(l_hid, num_units=rd,
                                      nonlinearity=lasagne.nonlinearities.rectify)
        l_hid = lasagne.layers.DropoutLayer(l_hid, p_dropout)

    if add_biases:
        l_hid = lasagne.layers.ConcatLayer([l_hid, l_biases], axis=-1)
        inputs.append(biases)

    #now B x 1        
    network = lasagne.layers.DenseLayer(l_hid, num_units=1,
                                             nonlinearity=T.nnet.sigmoid)

    predictions = lasagne.layers.get_output(network).ravel()

    #predictions = theano.tensor.log(predictions / (1 - predictions))
    #hloss = lasagne.objectives.binary_hinge_loss(predictions, gold) #, log_odds=False)
    hloss = lasagne.objectives.binary_crossentropy(predictions, gold)
    loss = lasagne.objectives.aggregate(hloss, weights, mode='normalized_sum')
    if topic and influence:
        loss += weighted_tloss + ortho_penalty
        
    all_params = lasagne.layers.get_all_params(l_recon, trainable=True)
    
    updates = lasagne.updates.adam(ntm_loss, all_params, learning_rate=lr)

    test_predictions = lasagne.layers.get_output(network, deterministic=True).ravel()

    print('rels...')    
    rels_fn = theano.function([in_words, in_dropmasks],
                              lasagne.layers.get_output(l_rels),
                              allow_input_downcast=True)
    print('train_ntm...')
    train_ntm_fn = theano.function([in_words, in_currmasks, in_dropmasks, in_neg, in_negmasks],
                                [ntm_loss, tloss, ortho_penalty],
                                updates=updates,
                                allow_input_downcast=True)
    print('train...')
    if topic and influence:
        all_params += lasagne.layers.get_all_params(network, trainable=True)
    else:
        all_params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=lr, momentum=0.9)
    
    if topic and influence:
        train_fn = theano.function([in_words, in_currmasks, in_dropmasks,
                                    in_neg, in_negmasks, idxs_rr, mask_rr_w, mask_rr_s,
                                    gold, weights, topic_weights, p_dropout],
                                    [loss, weighted_tloss, ortho_penalty, hloss],
                                    updates=updates,
                                    allow_input_downcast=True)
    else:
        train_fn = theano.function([idxs_rr, mask_rr_w, mask_rr_s,
                                    gold, weights, p_dropout],
                                    [loss, hloss],
                                    updates=updates,
                                    allow_input_downcast=True)
    print('predict...')
    if topic and influence:
        predict_fn = theano.function([in_words, in_dropmasks, idxs_rr, mask_rr_w, mask_rr_s],
                                    test_predictions,
                                    allow_input_downcast=True)
    else:
        predict_fn = theano.function([idxs_rr, mask_rr_w, mask_rr_s],
                                    test_predictions,
                                    allow_input_downcast=True)

    return train_fn, train_ntm_fn, rels_fn, predict_fn, l_recon, network, attention_layers

def save_ntm_params(layer, filename):
    params = lasagne.layers.get_all_params(layer)
    p_values = [p.get_value() for p in params]
    p_dict = dict(zip([str(p) for p in params], p_values))
    cPickle.dump(p_dict, open('{}_ntm_params.pkl'.format(filename), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    return p_dict

def save_inf_params(layer, filename):
    params = lasagne.layers.get_all_params(layer)
    p_values = [p.get_value() for p in params]
    p_dict = dict(zip(list(range(len(params))), p_values))
    cPickle.dump(p_dict, open('{}_inf_params.pkl'.format(filename), 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)
    return p_dict

def get_topic_neighbors(p_dict, We, filename, rev_indices):
    # compute nearest neighbors of descriptors
    R = p_dict['R']
    log = open(filename, 'w')
    for ind in range(len(R)):
        desc = R[ind] / np.linalg.norm(R[ind])
        sims = We.dot(desc.T)
        ordered_words = np.argsort(sims)[::-1]
        desc_list = [ rev_indices[w].encode('utf-8') for w in ordered_words[:10]]
        log.write(' '.join(desc_list) + '\n')
        print 'descriptor %d:' % ind
        print desc_list
    log.flush()
    log.close()

def get_top_attention_words(attention_layers, We, filename, rev_indices):
    log = open(filename, 'w')
    for i in attention_layers:
        if attention_layers[i] is None:
            continue
        #WD x HD
        W_w = attention_layers[0].W_w.get_value()
        #HD
        b_w = attention_layers[0].b_w.get_value()
        #K x HD
        u_w = attention_layers[0].u_w.get_value()
        
        if i == 0:
            u_w = [u_w]

        for ind in range(len(u_w)):
            #now V x HD
            h = np.tanh(We.dot(W_w) + b_w)
            sims = h.dot(u_w[ind])
            ordered_words = np.argsort(sims)[::-1]
            desc_list = [ rev_indices[w].encode('utf-8') for w in ordered_words[:10]]
            log.write(' '.join(desc_list) + '\n')
            print ('{} attention descriptor {}:'.format(i, ind))
            print (desc_list)
    log.flush()
    log.close()

def get_next_batch(idxs_batch, words, mask, words_rr, mask_rr, gold, num_negs, p_drop, deterministic=True):
        #op_idxs_batch = op_idxs[idxs_batch]
        words_batch = words[idxs_batch] #words[op_idxs_batch]
        mask_batch = mask[idxs_batch] #mask[op_idxs_batch]
        words_rr_batch = words_rr[idxs_batch]
        mask_rr_batch = mask_rr[idxs_batch]
        #make the sentence mask
        mask_rr_s_batch = (mask_rr_batch.sum(axis=-1) > 0).astype('float32')
        gold_batch = gold[idxs_batch]

        if deterministic:
            return words_batch, mask_batch, None, None, None, words_rr_batch, mask_rr_batch, mask_rr_s_batch, None, None

        ns, nm = utils.generate_negative_samples(words_batch.shape[0], num_negs,
                                           words.shape[1], words, mask)

        # word dropout
        # TODO: what if we drop words and there are no words left in a sentence
        drop_mask = (np.random.rand(*(mask_batch.shape)) < (1 - p_drop)).astype('float32')
        drop_mask *= mask_batch
        drop_mask_rr = (np.random.rand(*(mask_rr_batch.shape)) < (1 - p_drop)).astype('float32')
        drop_mask_rr *= mask_rr_batch
        #print(np.nonzero(drop_mask.sum(axis=-1))[0].shape)

        #calculate weights
        label_counts = collections.Counter(gold_batch)
        max_count = 1.*max(label_counts.values())
        class_weights = {i:1/(label_counts[i]/max_count) for i in label_counts}
        #print(label_counts, class_weights)
        weights = np.array([class_weights[i] for i in gold_batch]).astype(np.float32)

        return words_batch, mask_batch, drop_mask, ns, nm, words_rr_batch, drop_mask_rr, mask_rr_s_batch, gold_batch, weights

def main(data, indices, K=10, num_negs=10, lambda_t=1, num_epochs=15, batch_size=100, topic=False, influence=False, descriptor_log=None): 
    words = data['words']
    mask = data['mask']
    words_val = data['words_val']
    mask_val = data['mask_val']
    words_rr = data['words_rr']
    mask_rr = data['mask_rr']
    words_rr_val = data['words_rr_val']
    mask_rr_val = data['mask_rr_val']
    We = data['We']
    We_rr = data['We_rr']
    gold = data['gold']
    gold_val = data['gold_val']
    op_idxs = data['op_idxs']
    op_idxs_val = data['op_idxs_val']
                  
    mask_rr_s_val = (mask_rr_val.sum(axis=-1) > 0).astype('float32')
    
    norm_We = We / np.linalg.norm(We, axis=1)[:, None]
    We = np.nan_to_num(norm_We)

    # word dropout probability
    p_drop = 0.5 #0.75
    p_dropout = 0.25 #0.75
    
    lr = 0.05 #0.001
    eps = 1e-6
    rev_indices = {}
    for w in indices:
        rev_indices[indices[w]] = w
    for w in indices_rr:
        rev_indices_rr[indices_rr[w]] = w

    print 'compiling...'    
    train, train_ntm, get_topics, predict, ntm_layer, inf_layer, attention_layers = build_ntm(We.shape[1], We.shape[0],
                                               K, words.shape[1], We,
                                               words_rr.shape[1], words_rr.shape[2],
                                               We_rr.shape[0], We_rr,
                                               freeze_words=False,
                                               eps=eps, lr=lr, negs=num_negs, topic=topic,
                                               influence=influence,
                                               lambda_t=lambda_t)
    print 'done compiling, now training...'

    if descriptor_log is None:
        descriptor_log = 'descriptor_log'
    else:
        descriptor_log = descriptor_log
        
    print(np.setdiff1d(np.arange(mask.shape[0]),
                       np.nonzero(mask.sum(axis=-1))))

    #filter any OPs or RRs where there are no words
    '''
    mask_rr_batch = mask_rr[idxs_batch]
    op_idxs_batch = op_idxs[idxs_batch]
    mask_batch = mask[op_idxs_batch]
    print(mask_batch.sum(axis=-1).shape)
    print(mask_rr_batch.sum(axis=(1,2)).shape)
    print(np.nonzero(mask_batch.sum(axis=-1))[0].shape)
    valid_idxs_batch = np.intersect1d(np.nonzero(mask_batch.sum(axis=-1))[0],
                                      np.nonzero(mask_rr_batch.sum(axis=(1,2)))[0])
    idxs_batch = idxs_batch[valid_idxs_batch]
    print(len(idxs_batch), max(idxs_batch))
    '''
    words_op = words[op_idxs]
    mask_op = mask[op_idxs]
    valid_idxs = np.intersect1d(np.nonzero(mask_op.sum(axis=-1))[0],
                                np.nonzero(mask_rr.sum(axis=(1,2)))[0])
    print(words_op.shape, len(valid_idxs), max(valid_idxs))
    
    words = words_op[valid_idxs]
    mask = mask_op[valid_idxs]
    words_rr = words_rr[valid_idxs]
    mask_rr = mask_rr[valid_idxs]
    gold = gold[valid_idxs]
    
    # training loop
    min_cost = float('inf')
    max_val_score = 0
    num_batches = words_rr.shape[0] // batch_size + 1
    for epoch in range(num_epochs):
        cost = 0.
        cost_topic = 0.
        cost_inf = 0.
        idxs = np.random.choice(words_rr.shape[0], words_rr.shape[0], False)
        
        start_time = time.time()
        for batch_num in range(num_batches):
            print(batch_num)
            idxs_batch = idxs[batch_num*batch_size:(batch_num+1)*batch_size]
                        
            #op_idxs_batch = op_idxs[idxs_batch]
            words_batch = words[idxs_batch] #words[op_idxs_batch]
            mask_batch = mask[idxs_batch] #mask[op_idxs_batch]
            words_rr_batch = words_rr[idxs_batch]
            mask_rr_batch = mask_rr[idxs_batch]
            #make the sentence mask
            mask_rr_s_batch = (mask_rr_batch.sum(axis=-1) > 0).astype('float32')
            gold_batch = gold[idxs_batch]

            if len(gold_batch) < 1:
                continue
            ns, nm = utils.generate_negative_samples(words_batch.shape[0], num_negs,
                                               words.shape[1], words, mask)

            # word dropout
            # TODO: what if we drop words and there are no words left in a sentence
            drop_mask = (np.random.rand(*(mask_batch.shape)) < (1 - p_drop)).astype('float32')
            drop_mask *= mask_batch
            drop_mask_rr = (np.random.rand(*(mask_rr_batch.shape)) < (1 - p_drop)).astype('float32')
            drop_mask_rr *= mask_rr_batch
            #print(np.nonzero(drop_mask.sum(axis=-1))[0].shape)

            #calculate weights
            label_counts = collections.Counter(gold_batch)
            max_count = 1.*max(label_counts.values())
            class_weights = {i:1/(label_counts[i]/max_count) for i in label_counts}
            
            weights = np.array([class_weights[i] for i in gold_batch]).astype(np.float32)

            #words_batch, mask_batch, drop_mask, ns, nm, words_rr_batch, drop_mask_rr, mask_rr_s_batch, gold_batch, weights = get_next_batch(idxs_batch, words, mask, words_rr, mask_rr, gold)
            #topics = get_topics(words_batch, drop_mask)
            #print(topics)
            if influence:
                if topic:
                    ex_cost, ex_topic, ex_ortho, ex_inf = train(words_batch, mask_batch, drop_mask, ns, nm,
                                                                words_rr_batch, drop_mask_rr, mask_rr_s_batch,
                                                                gold_batch, weights, weights / num_negs,
                                                                p_dropout) #topic_weights
                    cost_topic += ex_topic                
                else:
                    ex_cost, ex_inf = train(words_rr_batch, drop_mask_rr, mask_rr_s_batch, gold_batch, weights, p_dropout)
                cost_inf += np.average(ex_inf, weights=weights)
            else:
                ex_cost, ex_topic, ex_ortho = train_ntm(words_batch, mask_batch, drop_mask, ns, nm)
                cost_topic += ex_topic
                
            cost += ex_cost
            
            #print(ex_cost, ex_topic, ex_ortho)
            if batch_num * batch_size % 1000 == 0:
                print(label_counts, class_weights)
                print(ex_cost, ex_topic if topic else None,
                      ex_ortho if topic else None,
                      np.average(ex_inf, weights=weights) if influence else None)
                print(time.time()-start_time)
            
        end_time = time.time()
        print(end_time-start_time, cost, cost_topic, cost_inf)
        
        #print predictions on validation set
        if influence:
            print(gold_val.shape)
            scores = []
            batch_size_val = gold_val.shape[0] // 50
            for i in range(gold_val.shape[0] // batch_size_val + 1):
                idxs_batch = np.arange(i*batch_size_val,min((i+1)*batch_size_val, gold_val.shape[0]))
                words_val_batch, mask_val_batch, _, _, _, words_rr_val_batch, mask_rr_val_batch, mask_rr_s_val_batch, _, _ = get_next_batch(idxs_batch, words_val[op_idxs_val], mask_val[op_idxs_val], words_rr_val, mask_rr_val, gold_val, num_negs, p_drop, True)
                if topic:
                    scores += predict(words_val_batch, mask_val_batch,
                                    words_rr_val_batch, mask_rr_val_batch, mask_rr_s_val_batch).tolist()
                else:
                    scores += predict(words_rr_val_batch, mask_rr_val_batch, mask_rr_s_val_batch).tolist()
                #scores += predict(words_val[op_idxs_val], mask_val[op_idxs_val],
                #                 words_rr_val, mask_rr_val, mask_rr_s_val).tolist()
            scores = np.nan_to_num(np.array(scores))
            predictions = scores > .5
            print(predictions.shape)
            val_score = roc_auc_score(gold_val, scores)
            print('ROC AUC for {},{}: {}'.format(K, lambda_t, val_score))
            precision, recall, fscore, _ = precision_recall_fscore_support(gold_val, predictions)
            print('Precision for {},{}: {} Recall: {} F1: {}'.format(K, lambda_t, precision, recall, fscore))
            print('Accuracy for {},{}: {}'.format(K, lambda_t, accuracy_score(gold_val, predictions)))

            #TODO: get modified We and We_rr if freeze is false
            if val_score < max_val_score:
                max_val_score = val_score
                save_inf_params(inf_layer, descriptor_log + '_inf')
                get_top_attention_words(attention_layers, We_rr, descriptor_log + '_inf', rev_indices_rr)

                if topic:
                    p_dict = save_ntm_params(ntm_layer, descriptor_log + '_inf')
                    get_topic_neighbors(p_dict, We, descriptor_log + '_inf', rev_indices)

        # save params if cost went down
        if cost < min_cost:
            if topic:
                min_cost = cost
                save_ntm_params(ntm_layer, descriptor_log + '_ntm')
                get_topic_neighbors(p_dict, We, descriptor_log + '_ntm', rev_indices)
                if influence:
                    save_inf_params(inf_layer, descriptor_log + '_ntm')
                    get_top_attention_words(attention_layers, We_rr, descriptor_log + '_ntm', rev_indices_rr)

            #rels = get_topics(words, mask)
            #print(rels.sum(axis=0))
            #print(collections.Counter(rels.argmax(axis=1)))

            '''
            #save influence classifier parameters
            params = lasagne.layers.get_all_params(inf_layer)
            p_values = [p.get_value() for p in params]
            cPickle.dump(p_values, open('inf_params.pkl', 'wb'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
            #TODO
            #compute highest scoring attention words per topic
            '''
            
        print 'done with epoch: ', epoch, ' cost =',\
            cost / words_batch.shape[0], 'time: ', end_time-start_time
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='train an influence classifier')
    parser.add_argument('metadata_file')
    parser.add_argument('embeddings_file')
    
    parser.add_argument('--min_count', type=int, default=0),
    parser.add_argument('--max_count', type=int, default=2**32),
    parser.add_argument('--min_rank', type=int, default=0),
    parser.add_argument('--max_rank', type=int, default=1),
    
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--num_negs', type=int, default=10)

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--topic', action='store_true')
    parser.add_argument('--influence', action='store_true')    
    parser.add_argument('--lambda_t', type=float, default=1.0)
    parser.add_argument('--load')
    parser.add_argument('--save')
    parser.add_argument('--descriptor_log')
            
    args = parser.parse_args()

    print('loading data...')
    if not args.load:
        with open(args.metadata_file) as f:
            metadata = json.load(f)

        embeddings = gensim.models.Doc2Vec.load_word2vec_format(args.embeddings_file, binary=False)

        
        words, mask, indices, counts, _ = utils.load_data(metadata, embeddings,
                                                   args.min_count, args.max_count,
                                                   args.min_rank, args.max_rank,
                                                   add=True)
        words_val, mask_val, _, _, We = utils.load_data(metadata, embeddings,
                                                   args.min_count, args.max_count,
                                                   args.min_rank, args.max_rank,
                                                   indices=indices,
                                                   add=True,
                                                   counts=counts,
                                                   keys=['val_op'])

        words_rr, mask_rr, indices_rr, counts_rr, We_rr = utils.load_data(metadata, embeddings,
                                                                          args.min_count, args.max_count,
                                                                          args.min_rank, args.max_rank,
                                                                          add=True,
                                                                          hierarchical=True,
                                                                          keys=['train_pos', 'train_neg'])
        words_rr_val, mask_rr_val, _, _, _ = utils.load_data(metadata, embeddings, 
                                                                          args.min_count, args.max_count,
                                                                          args.min_rank, args.max_rank,
                                                                          indices=indices_rr,
                                                                          add=False,
                                                                          counts=counts_rr,
                                                                          hierarchical=True,
                                                                          keys=['val_pos', 'val_neg'])
        
        op_idxs = np.array(metadata['train_pos_indices'] + metadata['train_neg_indices'])
        gold = np.array([1]*len(metadata['train_pos']) + [0]*len(metadata['train_neg']))
        op_idxs_val = np.array(metadata['val_pos_indices'] + metadata['val_neg_indices'])
        gold_val = np.array([1]*len(metadata['val_pos']) + [0]*len(metadata['val_neg']))
        
        if args.save:
            with open(args.save + '_indices.json', 'w') as f:
                json.dump([indices, indices_rr], f)
            np.savez(args.save, words=words, mask=mask, words_val=words_val,
                     mask_val=mask_val, words_rr=words_rr, mask_rr=mask_rr,
                     words_rr_val=words_rr_val, mask_rr_val=mask_rr_val,
                     We=We, We_rr=We_rr,
                     gold=gold, gold_val=gold_val, op_idxs=op_idxs, op_idxs_val=op_idxs_val)
            data=np.load(args.save + '.npz')
    else:
        with open(args.load + '_indices.json') as f:
            indices, indices_rr = json.load(f)

        data = np.load(args.load + '.npz')

    if args.topic:
        Ks = [10, 25, 50]
        lambda_ts = [1, .1, .01, .001, .0001, .00001]
    else:
        Ks = [1]
        lambda_ts = [1]
    for K in Ks:
        for lambda_t in lambda_ts:
            main(data, indices, indices_rr, K, args.num_negs, lambda_t, args.num_epochs, args.batch_size, args.topic, args.influence, args.descriptor_log)
