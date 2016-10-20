import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer

class ArgumentationRNN:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False):

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

        #l_idxs_drop_op = lasagne.layers.DropoutLayer(l_idxs_op, .75)
        #l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_drop_op, V, d,
        #                                           W=lasagne.utils.floatX(embeddings))
        
        #now B x S x N x D
        l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d, W=l_emb_op_w.W)

        #now average over word embeddings in each sentence
        #alternatively, feed into an LSTM first
        #alternatively, calculate the attention over words
        #now B x S x D

        #add dropout layer
        #l_drop_emb_op_w = lasagne.layers.DropoutLayer(l_emb_op_w, .2)
        #l_avg_op_s = AverageWordLayer([l_drop_emb_op_w, l_mask_op_w, l_mask_op_s])

        #CBOW w/attn
        l_attn_op_w = AttentionWordLayer([l_emb_op_w, l_mask_op_w], d)
        l_avg_op_s = WeightedAverageWordLayer([l_emb_op_w, l_attn_op_w])
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        #LSTM w/attn
        #need to reshape input
        '''
        l_reshape_op_w = lasagne.layers.ReshapeLayer(l_emb_op_w,
                                                     (idxs_op.shape[0]*max_post_length,
                                                    max_sentence_length, d))
        l_reshape_mask_op_w = lasagne.layers.ReshapeLayer(l_mask_op_w,
                                                          (idxs_op.shape[0]*max_post_length,
                                                         max_sentence_length))
        l_lstm_op_w = lasagne.layers.LSTMLayer(l_reshape_op_w, d,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            grad_clipping=GRAD_CLIP,
                                            mask_input=l_reshape_mask_op_w)
        l_reshape_lstm_op_w = lasagne.layers.ReshapeLayer(l_lstm_op_w,
                                                          (idxs_op.shape[0], max_post_length,
                                                         max_sentence_length, d))
        '''
        #shape = (idxs_op.shape[0], max_post_length, max_sentence_length, d)
        #l_reshape_lstm_op_w = reshapeLSTMLayer(l_emb_op_w, l_mask_op_w, shape, d)
        #l_attn_op_w = AttentionWordLayer([l_reshape_lstm_op_w, l_mask_op_w], d)
        #l_avg_op_s = WeightedAverageWordLayer([l_reshape_lstm_op_w, l_attn_op_w])

        #l_avg_op_s = AverageWordLayer([l_emb_op_w, l_mask_op_w])
        #l_avg_rr_s = AverageWordLayer([l_emb_rr_w, l_mask_rr_w])        

        #now use this as an input to an LSTM
        #shape is still B x S x D
        l_lstm_op_s = lasagne.layers.LSTMLayer(l_avg_op_s, rd,
                                           nonlinearity=lasagne.nonlinearities.tanh,
        #l_lstm_op_s = lasagne.layers.GRULayer(l_avg_op_s, rd,
                                            grad_clipping=GRAD_CLIP,
                                            mask_input=l_mask_op_s)
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)#,
        '''
                                               hid_init=lasagne.init.Constant(0.),
                                               ingate=lasagne.layers.Gate(W_in=l_lstm_op_s.W_in_to_ingate,
                                                                          W_hid=l_lstm_op_s.W_hid_to_ingate,
                                                                          W_cell=l_lstm_op_s.W_cell_to_ingate,
                                                                          b=l_lstm_op_s.b_ingate),
                                               outgate=lasagne.layers.Gate(W_in=l_lstm_op_s.W_in_to_outgate,
                                                                           W_hid=l_lstm_op_s.W_hid_to_outgate,
                                                                           W_cell=l_lstm_op_s.W_cell_to_outgate,
                                                                           b=l_lstm_op_s.b_outgate),
                                               forgetgate=lasagne.layers.Gate(W_in=l_lstm_op_s.W_in_to_forgetgate,
                                                                              W_hid=l_lstm_op_s.W_hid_to_forgetgate,
                                                                              W_cell=l_lstm_op_s.W_cell_to_forgetgate,
                                                                              b=l_lstm_op_s.b_forgetgate),
                                               cell=lasagne.layers.Gate(W_in=l_lstm_op_s.W_in_to_cell,
                                                                        W_hid=l_lstm_op_s.W_hid_to_cell,
                                                                        W_cell=None,
                                                                        b=l_lstm_op_s.b_cell,
                                                                        nonlinearity=lasagne.nonlinearities.tanh))
        '''
                                                                          
        #l_lstm_rr_s = lasagne.layers.GRULayer(l_avg_rr_s, rd,
        #                                      grad_clipping=GRAD_CLIP,
        #                                      mask_input=l_mask_rr_s,
        #                                      )
        
        
        #now average over sentences
        #alternatively, calculate attention over words
        #shape is now B x D
        #l_drop_lstm_op_s = lasagne.layers.DropoutLayer(l_lstm_op_s, .5)
        #l_lstm_op_avg = AverageSentenceLayer([l_drop_lstm_op_s, l_mask_op_s])
        
        #l_lstm_op_avg = AverageSentenceLayer([l_lstm_op_s, l_mask_op_s])

        #LSTM w/ attn
        l_attn_op_s = AttentionSentenceLayer([l_lstm_op_s, l_mask_op_s], rd)        
        l_lstm_op_avg = WeightedAverageSentenceLayer([l_lstm_op_s, l_attn_op_s])
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])

        #no LSTM, attn
        #l_attn_op_s = AttentionSentenceLayer([l_avg_op_s, l_mask_op_s], d)        
        #l_lstm_op_avg = WeightedAverageSentenceLayer([l_avg_op_s, l_attn_op_s])
                
        #now B x 2RD
        #l_merge_op_rr = lasagne.layers.ConcatLayer([l_lstm_op_avg, l_lstm_rr_avg])
        #l_hid1 = lasagne.layers.DenseLayer(l_merge_op_rr, num_units=rd,
        #                                  nonlinearity=lasagne.nonlinearities.rectify)
        #self.network = lasagne.layers.DenseLayer(l_hid1, num_units=1,
        #                                         nonlinearity=T.nnet.sigmoid)

        #TODO: add another hidden layer
        #add difference and elemwise mult of layers
        #add some GRU type weighting (1-alpha)*OP + alpha*RR where alpha is learned
        
        #now B x RD
        #self.network = lasagne.layers.DenseLayer(l_lstm_op_avg, num_units=1,
        #                                  nonlinearity=T.nnet.sigmoid)

        #bilinear
        #still B x RD
        #l_hid1 = lasagne.layers.DenseLayer(l_lstm_op_avg, num_units=rd,
        #                                  nonlinearity=lasagne.nonlinearities.linear)
        #still B x RD
        #self.network = lasagne.layers.ElemwiseMergeLayer([l_hid1, l_lstm_rr_avg], T.mul)

        #add another RNN with the OP and RR
        l_merge_op_rr = lasagne.layers.ConcatLayer([l_lstm_op_avg, l_lstm_rr_avg])
        l_reshape_op_rr = lasagne.layers.ReshapeLayer(l_merge_op_rr,
                                                      (idxs_op.shape[0], 2, rd))
        l_gru_op_rr = lasagne.layers.GRULayer(l_reshape_op_rr, rd, grad_clipping=GRAD_CLIP)
        
        #add attention layer? original point of view or response more highly weighted?
        l_attn_op_rr = AttentionSentenceLayer([l_gru_op_rr], rd)
        #now B x 2RD
        l_gru_op_rr_avg = WeightedAverageSentenceLayer([l_gru_op_rr, l_attn_op_rr])

        l_reshape_gru_op_rr = lasagne.layers.ReshapeLayer(l_gru_op_rr,
                                                          (idxs_op.shape[0], 2*rd))
        l_hid1 = lasagne.layers.DenseLayer(l_reshape_gru_op_rr, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)
        self.network = lasagne.layers.DenseLayer(l_hid1, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)

        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #add regularization
        lambda_w = T.scalar('lambda_w')
        loss += lambda_w*apply_penalty(params, l2)

        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=0.01, momentum=0.9)

        print('compiling...')
        self.train = theano.function([idxs_op, idxs_rr,
                                      mask_op_w, mask_rr_w,
                                      mask_op_s, mask_rr_s,
                                      gold, lambda_w],
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
                                         gold, lambda_w],
                                        [loss, test_acc])

        #attention for words, B x S x N
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
                
        print('finished compiling...')

    def save(self, filename):
        param_values = lasagne.layers.get_all_param_values(self.network)
        np.savez_compressed(filename, *param_values)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
