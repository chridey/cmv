import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, BroadcastLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer, LSTMDecoderLayer

#theano.config.compute_test_value = 'off' 

class ArgumentationEncoderDecoderRNN:
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

        #need a B x 1 index and mask layer
        l_eop = lasagne.layers.InputLayer(shape=(None, 1),
                                          input_var=T.zeros((idxs_op.shape[0], 1),
                                                            dtype='int32'))
        l_mask_eop = lasagne.layers.InputLayer(shape=(None, 1),
                                               input_var=T.ones((idxs_op.shape[0], 1),
                                                                dtype='int32'))
        
        #now B x 1 x D
        l_emb_eop = lasagne.layers.EmbeddingLayer(l_eop, V, d,
                                                  W=l_emb_op_w.W)
        #now concat op, eop, and rr
        #B x (2S + 1) x D
        l_thread = lasagne.layers.ConcatLayer([l_avg_op_s, l_emb_eop, l_avg_rr_s], axis=1)
        #B x (2S + 1)
        l_mask_thread = lasagne.layers.ConcatLayer([l_mask_op_s, l_mask_eop, l_mask_rr_s], axis=1)
        l_lstm_thread = lasagne.layers.LSTMLayer(l_thread, rd,
                                                 nonlinearity=lasagne.nonlinearities.tanh,
                                                 grad_clipping=GRAD_CLIP,
                                                 mask_input=l_mask_thread)
        #lstm_op and lstm_rr are the first S and last S in the lstm
        l_lstm_op_s = lasagne.layers.SliceLayer(l_lstm_thread,
                                                #indices=max_post_length, #
                                                slice(0, max_post_length),
                                                axis=1)
        l_lstm_rr_s = lasagne.layers.SliceLayer(l_lstm_thread,
                                                #indices=-1, #
                                                slice(-max_post_length, None),
                                                axis=1)
        
        #LSTM w/ attn
        '''
        l_lstm_op_s = lasagne.layers.LSTMLayer(l_avg_op_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_op_s)

        l_attn_op_s = AttentionSentenceLayer([l_lstm_op_s, l_mask_op_s], rd)        
        l_lstm_op_avg = WeightedAverageSentenceLayer([l_lstm_op_s, l_attn_op_s])

        l_lstm_rr_s = LSTMDecoderLayer([l_avg_rr_s, l_lstm_op_avg], rd,
                                       #hid_init=lasagne.layers.SliceLayer(l_lstm_op_s,
                                       #                                   indices=-1,
                                       #                                   axis=1), #l_lstm_op_avg,
                                       nonlinearity=lasagne.nonlinearities.tanh,
                                       grad_clipping=GRAD_CLIP,
                                       mask_input=l_mask_rr_s)
        '''
        
        #attention is dependent on the response as well as the OP
        #if we are not doing alignments, we need to broadcast the B x D context layer to B x S x D
        '''
        l_lstm_op_avg_context = l_lstm_op_avg
        if not alignment:
            l_lstm_op_avg_context = BroadcastLayer([l_lstm_op_avg, 
                                                   l_lstm_rr_s])
        print(l_lstm_op_avg_context.output_shape)
        print(l_lstm_rr_s.output_shape)

        l_concat_lstm_rr_op_avg = lasagne.layers.ConcatLayer([l_lstm_rr_s,
                                                              l_lstm_op_avg_context],
                                                              axis=-1)
        
        l_attn_rr_s = AttentionSentenceLayer([l_concat_lstm_rr_op_avg, l_mask_rr_s], rd)

        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               #hid_init=lasagne.layers.SliceLayer(l_lstm_op_s,
                                               #                                   indices=-1,
                                               #                                   axis=1),
                                               #hid_init=l_lstm_op_avg,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)


        '''                                               
        #l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        #l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        #l_attn_op_s = AttentionSentenceLayer([l_lstm_op_s, l_mask_op_s], rd)        
        #l_lstm_op_avg = WeightedAverageSentenceLayer([l_lstm_op_s, l_attn_op_s])

        #l_lstm_rr_avg = AverageSentenceLayer([l_lstm_rr_s, l_mask_rr_s])
        #l_lstm_op_avg = AverageSentenceLayer([l_lstm_op_s, l_mask_op_s])
        
        l_lstm_rr_avg = lasagne.layers.SliceLayer(l_lstm_rr_s, indices=-1, axis=1)
        l_lstm_op_avg = lasagne.layers.SliceLayer(l_lstm_op_s, indices=-1, axis=1)
        
        #concatenate with rr-op and rr*op
        l_diff = lasagne.layers.ElemwiseMergeLayer([l_lstm_rr_avg, l_lstm_op_avg],
                                                   T.sub)
        l_mul = lasagne.layers.ElemwiseMergeLayer([l_lstm_rr_avg, l_lstm_op_avg],
                                                   T.mul)
        #l_concat = lasagne.layers.ConcatLayer([l_lstm_rr_avg, l_diff, l_mul], axis=-1)
        if diff:
            l_concat = lasagne.layers.ConcatLayer([l_lstm_rr_avg, l_diff], axis=-1)
        else:
            l_concat = l_lstm_rr_avg

        #l_hid1 = lasagne.layers.DenseLayer(l_lstm_rr_avg, num_units=rd,
        l_hid1 = lasagne.layers.DenseLayer(l_concat, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)
        l_hid1 = lasagne.layers.DropoutLayer(l_hid1, p_dropout)
        self.network = lasagne.layers.DenseLayer(l_hid1, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold).mean()
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
        
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
                                      gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True)
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
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
                                         gold, lambda_w, p_dropout],
                                        [loss, test_acc])

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
        
