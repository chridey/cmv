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
from cmv.rnn.layers import TopicAttentionWordLayer, WeightedAverageWordLayer, AttentionSentenceLayer, WeightedAverageSentenceLayer



# assemble the network
def build_rmn(d_word, len_voc, 
    num_descs, max_len, We,
    max_post_length, max_sentence_length,
    len_voc_rr, We_rr, rd=50,
    freeze_words=True, eps=1e-5, lr=0.01, negs=10,
    num_layers=2, add_biases=False,
    GRAD_CLIP=100):

    # input theano vars
    in_words = T.imatrix(name='words')
    in_neg = T.itensor3(name='neg')
    in_currmasks = T.matrix(name='curr_masks')
    in_dropmasks = T.matrix(name='drop_masks')
    in_negmasks = T.itensor3(name='neg_masks')

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
        d_word, W=We, name='word_emb')
    # negative examples should use same embedding matrix
    # B x N x L x D
    l_negemb = MyEmbeddingLayer(l_inneg, len_voc, 
            d_word, W=l_emb.W, name='word_emb_copy1')

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

    tloss = T.sum(T.maximum(0., 1. - correct[:, None] + negs))
    #TODO: add normalized sum, so that loss is of same magnitude
    
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
    weights = T.ivector('weights')

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
    l_attn_rr_w = TopicAttentionWordLayer([l_emb_rr_w, l_mask_rr_w, l_rels], d_word)
    l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])
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

    predictions = theano.tensor.log(predictions / (1 - predictions))
    hloss = lasagne.objectives.binary_hinge_loss(predictions, gold) #, log_odds=False)
    loss = ntm_loss + lasagne.objectives.aggregate(hloss, weights, mode='normalized_sum')

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
    all_params += lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)    
    train_fn = theano.function([in_words, in_currmasks, in_dropmasks,
                                in_neg, in_negmasks, idxs_rr, mask_rr_w, mask_rr_s,
                                gold, weights, p_dropout],
                                [loss, tloss, ortho_penalty, hloss],
                                updates=updates,
                                allow_input_downcast=True)
    
    print('predict...')
    predict_fn = theano.function([in_words, in_dropmasks, idxs_rr, mask_rr_w, mask_rr_s],
                                 test_predictions,
                                 allow_input_downcast=True)

    return train_fn, train_ntm_fn, rels_fn, predict_fn, l_recon, network

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
    
    args = parser.parse_args()

    print('loading data...')
    with open(args.metadata_file) as f:
        metadata = json.load(f)

    embeddings = gensim.models.Doc2Vec.load_word2vec_format(args.embeddings_file, binary=False)
    
    words, mask, indices, counts, We = utils.load_data(metadata, embeddings,
                                               args.min_count, args.max_count,
                                               args.min_rank, args.max_rank,
                                               add=True)
    words_val, mask_val, _, _, _ = utils.load_data(metadata, embeddings,
                                               args.min_count, args.max_count,
                                               args.min_rank, args.max_rank,
                                               indices=indices,
                                               add=False,
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
    mask_rr_s_val = (mask_rr_val.sum(axis=-1) > 0).astype('float32')
    
    norm_We = We / np.linalg.norm(We, axis=1)[:, None]
    We = np.nan_to_num(norm_We)

    # word dropout probability
    p_drop = 0.75
    p_dropout = 0.75
    
    lr = 0.001
    eps = 1e-6
    rev_indices = {}
    for w in indices:
        rev_indices[indices[w]] = w

    print 'compiling...'    
    train, train_ntm, get_topics, predict, ntm_layer, inf_layer = build_rmn(We.shape[1], We.shape[0],
                                               args.K, words.shape[1], We,
                                               words_rr.shape[1], words_rr.shape[2],
                                               We_rr.shape[0], We_rr,
                                               freeze_words=True,
                                               eps=eps, lr=lr, negs=args.num_negs)
    print 'done compiling, now training...'
    
    descriptor_log = 'descriptor_log'
    op_idxs = np.array(metadata['train_pos_indices'] + metadata['train_neg_indices'])
    gold = np.array([1]*len(metadata['train_pos']) + [0]*len(metadata['train_neg']))
    op_idxs_val = np.array(metadata['val_pos_indices'] + metadata['val_neg_indices'])
    gold_val = np.array([1]*len(metadata['val_pos']) + [0]*len(metadata['val_neg']))

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
    num_batches = words_rr.shape[0] // args.batch_size + 1
    for epoch in range(args.num_epochs):
        cost = 0.
        idxs = np.random.choice(words_rr.shape[0], words_rr.shape[0], False)
        
        start_time = time.time()
        for batch_num in range(num_batches):
            print(batch_num)
            idxs_batch = idxs[batch_num*args.batch_size:(batch_num+1)*args.batch_size]
                        
            #op_idxs_batch = op_idxs[idxs_batch]
            words_batch = words[idxs_batch] #words[op_idxs_batch]
            mask_batch = mask[idxs_batch] #mask[op_idxs_batch]
            words_rr_batch = words_rr[idxs_batch]
            mask_rr_batch = mask_rr[idxs_batch]
            #make the sentence mask
            mask_rr_s_batch = (mask_rr_batch.sum(axis=-1) > 0).astype('float32')
            gold_batch = gold[idxs_batch]
            
            ns, nm = utils.generate_negative_samples(words_batch.shape[0], args.num_negs,
                                               words.shape[1], words, mask)

            # word dropout
            # TODO: what if we drop words and there are no words left in a sentence
            drop_mask = (np.random.rand(*(mask_batch.shape)) < (1 - p_drop)).astype('float32')
            drop_mask *= mask_batch
            drop_mask_rr = (np.random.rand(*(mask_rr_batch.shape)) < (1 - p_drop)).astype('float32')
            drop_mask_rr *= mask_rr_batch
            print(np.nonzero(drop_mask.sum(axis=-1))[0].shape)
            
            #calculate weights
            label_counts = collections.Counter(gold_batch)
            max_count = 1.*max(label_counts.values())
            class_weights = {i:1/(label_counts[i]/max_count) for i in label_counts}
            print(label_counts, class_weights)
            weights = np.array([class_weights[i] for i in gold_batch]).astype(np.float32)

            #topics = get_topics(words_batch, drop_mask)
            #print(topics)
            ex_cost, ex_topic, ex_ortho, ex_inf = train(words_batch, mask_batch, drop_mask, ns, nm,
                                                        words_rr_batch, drop_mask_rr, mask_rr_s_batch,
                                                        gold_batch, weights, p_dropout) #topic_weights
            #ex_cost, ex_topic, ex_ortho = train_ntm(words_batch, mask_batch, drop_mask, ns, nm)
            
            cost += ex_cost
            print(ex_cost, ex_topic, ex_ortho, ex_inf.sum())
            #print(ex_cost, ex_topic, ex_ortho)
            
        end_time = time.time()
        print(cost)
        
        #print predictions on validation set
        print(gold_val.shape)
        scores = predict(words_val[op_idxs_val], mask_val[op_idxs_val],
                              words_rr_val, mask_rr_val, mask_rr_s_val)
        scores = np.nan_to_num(scores)
        predictions = scores > .5
        print(predictions.shape)
        print('ROC AUC', roc_auc_score(gold_val, scores))
        print('Precision: {} Recall: {} F1: {}'.format(*(precision_recall_fscore_support(gold_val, predictions)[:3])))
        print('Accuracy: ', accuracy_score(gold_val, predictions))
            
        # save params if cost went down
        if cost < min_cost:
            min_cost = cost
            params = lasagne.layers.get_all_params(ntm_layer)
            p_values = [p.get_value() for p in params]
            p_dict = dict(zip([str(p) for p in params], p_values))
            cPickle.dump(p_dict, open('ntm_params.pkl', 'wb'),
                protocol=cPickle.HIGHEST_PROTOCOL)

            # compute nearest neighbors of descriptors
            R = p_dict['R']
            log = open(descriptor_log, 'w')
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
            rels = get_topics(words, mask)
            print(rels.sum(axis=0))
            print(collections.Counter(rels.argmax(axis=1)))

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
    
