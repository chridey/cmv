import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, BroadcastLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer, LSTMDecoderLayer

#theano.config.compute_test_value = 'off' 

class ArgumentationBeliefState:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 alignment=False,
                 shared=True,
                 diff=False):

        print('using belief state...')
        
        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_op = T.itensor3('idxs_op') #imatrix
        idxs_rr = T.itensor3('idxs_rr') #imatrix
        #B x S x N matrix
        mask_op_w = T.itensor3('mask_op_w')
        mask_rr_w = T.itensor3('mask_rr_w')
        #B x S matrix
        mask_op_s = T.imatrix('mask_op_s')
        mask_rr_s = T.imatrix('mask_rr_s')
        #B-long vector
        gold = T.ivector('gold')
        p_dropout = T.scalar('p_dropout')
                
        #now use this as an input to an LSTM
        l_idxs_op = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_op)
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_mask_op_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_op_w)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_op_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_op_s)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)

        
        #now B x S x N x D
        l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        if shared:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d, W=l_emb_op_w.W)
        else:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                       W=lasagne.utils.floatX(embeddings))

        #CBOW w/attn
        l_attn_op_w = AttentionWordLayer([l_emb_op_w, l_mask_op_w], d)
        l_avg_op_s = WeightedAverageWordLayer([l_emb_op_w, l_attn_op_w])
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        #now use this as an input to an LSTM
        #shape is still B x S x D

        #LSTM w/ attn
        l_lstm_op_s = lasagne.layers.LSTMLayer(l_avg_op_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_op_s)

        l_attn_op_s = AttentionSentenceLayer([l_lstm_op_s, l_mask_op_s], rd)        
        l_lstm_op_avg = WeightedAverageSentenceLayer([l_lstm_op_s, l_attn_op_s])

        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)

        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])

        #shape is now B x RD, then B x 2*RD, then B x 2 x RD
        l_lstm_avg = lasagne.layers.ConcatLayer([l_lstm_op_avg, l_lstm_rr_avg], axis=-1)
        l_lstm_avg = lasagne.layers.ReshapeLayer(l_lstm_avg, shape=([0], 2, rd))
        
        #shape is still B x 2 x RD                                         
        l_belief_gru = lasagne.layers.GRULayer(l_lstm_avg,
                                               rd,
                                               grad_clipping=GRAD_CLIP)
        
        #dimshuffle to 2 x B x RD, so that all OP are along the first axis
        l_dimshuffle = lasagne.layers.DimshuffleLayer(l_belief_gru,
                                                      (1,0,2))
        #max_responses
        l_concat = lasagne.layers.ReshapeLayer(l_dimshuffle,
                                               (-1,
                                                rd))                                               
        #l_concat = lasagne.layers.SliceLayer(l_belief_gru,
        #                                     -1,
        #                                     1)
        
        #l_hid1 = lasagne.layers.DenseLayer(l_lstm_rr_avg, num_units=rd,
        l_hid1 = lasagne.layers.DenseLayer(l_concat, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)
        l_hid1 = lasagne.layers.DropoutLayer(l_hid1, p_dropout)
        self.network = lasagne.layers.DenseLayer(l_hid1, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        loss = lasagne.objectives.binary_hinge_loss(predictions, gold).mean()
        gold_plus = T.concatenate([T.zeros_like(gold), gold])
        datum_loss = lasagne.objectives.binary_crossentropy(predictions, gold_plus)
        weight = T.scalar('weight')
        weights = T.concatenate([weight*T.ones_like(gold, dtype=theano.config.floatX),
                                 T.ones_like(gold, dtype=theano.config.floatX)])
        loss = lasagne.objectives.aggregate(datum_loss, weights, 'normalized_sum')
        #loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #add regularization
        lambda_w = T.scalar('lambda_w')
        loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.adam(loss, params)
        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=0.01, momentum=0.9)

        print('compiling...')
        self.train = theano.function([idxs_op, idxs_rr,
                                      mask_op_w, mask_rr_w,
                                      mask_op_s, mask_rr_s,
                                      gold, lambda_w, p_dropout, weight],
                                     loss, updates=updates, allow_input_downcast=True,
                                     on_unused_input='warn')
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        test_predictions = test_predictions[test_predictions.shape[0]/2:]
        self.predict = theano.function([idxs_op, idxs_rr,
                                        mask_op_w, mask_rr_w,
                                        mask_op_s, mask_rr_s,
                                        ],
                                       test_predictions, allow_input_downcast=True)

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        test_loss = lasagne.objectives.binary_crossentropy(test_predictions, gold).mean()        
        self.validate = theano.function([idxs_op, idxs_rr,
                                         mask_op_w, mask_rr_w,
                                         mask_op_s, mask_rr_s,
                                         gold, lambda_w, p_dropout, weight],
                                        [loss, test_acc],
                                        on_unused_input='warn')

        #attention for words, B x S x N
        '''
        word_attention = lasagne.layers.get_output(l_attn_op_w)
        self.word_attention = theano.function([idxs_op,
                                               mask_op_w],
                                               word_attention, allow_input_downcast=True)
            
        #attention for sentences, B x S
        sentence_attention = lasagne.layers.get_output(l_attn_op_s)
        self.sentence_attention = theano.function([idxs_op,
                                                mask_op_w,
                                                mask_op_s],
                                                sentence_attention, allow_input_downcast=True)
        '''
        print('finished compiling...')

    def save(self, filename):
        param_values = lasagne.layers.get_all_param_values(self.network)
        np.savez_compressed(filename, *param_values)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
