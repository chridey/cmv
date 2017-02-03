import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer
from cmv.rnn.loss import margin_loss

class ArgumentationRNN:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 num_layers=1,
                 learning_rate=0.01,
                 d_title=0):

        print(d_title)
        
        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_rr = T.itensor3('idxs_rr') #imatrix
        #B x S x N matrix
        mask_rr_w = T.itensor3('mask_rr_w')
        #B x S matrix
        mask_rr_s = T.imatrix('mask_rr_s')
        #B-long vector
        gold = T.ivector('gold')
        lambda_w = T.scalar('lambda_w')
        p_dropout = T.scalar('p_dropout')

        idxs_rr_title = T.imatrix('idxs_title_rr')
        mask_rr_title = T.imatrix('mask_rr_title')
        
        #now use this as an input to an LSTM
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)
        l_idxs_rr_title = lasagne.layers.InputLayer(shape=(None, max_sentence_length),
                                            input_var=idxs_rr_title)
        l_mask_rr_title = lasagne.layers.InputLayer(shape=(None, max_sentence_length),
                                            input_var=mask_rr_title)
        
        #now B x S x N x D
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        l_emb_rr_title = lasagne.layers.EmbeddingLayer(l_idxs_rr_title, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        '''
        l_reshape_rr_w = lasagne.layers.ReshapeLayer(l_emb_rr_w, (idxs_rr.shape[0],
                                                                  max_post_length*max_sentence_length,
                                                                  d))
        l_reshape_mask_rr_w = lasagne.layers.ReshapeLayer(l_mask_rr_w, (idxs_rr.shape[0],
                                                                    max_post_length*max_sentence_length))
    
        l_lstm_rr_w = lasagne.layers.LSTMLayer(l_reshape_rr_w, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_reshape_mask_rr_w)
        l_lstm_rr_w = lasagne.layers.ReshapeLayer(l_lstm_rr_w, (idxs_rr.shape[0],
                                                                max_post_length,
                                                                max_sentence_length,
                                                                rd))
        l_attn_rr_w = AttentionWordLayer([l_lstm_rr_w, l_mask_rr_w], rd)
        l_avg_rr_s = WeightedAverageWordLayer([l_lstm_rr_w, l_attn_rr_w])
        #l_avg_rr_s = AverageWordLayer([l_lstm_rr_w, l_mask_rr_w])
        #l_avg_rr_s = lasagne.layers.SliceLayer(l_lstm_rr_w, indices=-1, axis=2)
        '''
        
        #CBOW w/attn
        #now B x S x D
        custom_query = None
        d_attn = d
        if d_title:
            l_rr_title_s = lasagne.layers.LSTMLayer(l_emb_rr_title, d_title,
                                                    nonlinearity=lasagne.nonlinearities.tanh,
                                                    grad_clipping=GRAD_CLIP,
                                                    mask_input=l_mask_rr_title)
            l_rr_title_s = lasagne.layers.SliceLayer(l_rr_title_s, indices=-1, axis=1)
            custom_query = l_rr_title_s
            d_attn = d_title
            
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d_attn,
                                         custom_query=custom_query)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        #CHANGED
        #now BS x D
        '''
        l_reshape_s = lasagne.layers.ReshapeLayer(l_avg_rr_s, (idxs_rr.shape[0]*max_post_length, d))
        #now BS x RD
        l_hid1_s = lasagne.layers.DenseLayer(l_reshape_s, num_units=rd,
                                            nonlinearity=lasagne.nonlinearities.rectify)
        l_drop_s = l_hid1_s #l_drop_s = lasagne.layers.DropoutLayer(l_hid1_s, p_dropout)
        l_avg_rr_s = lasagne.layers.DenseLayer(l_drop_s, num_units=rd,
                                                 nonlinearity=lasagne.nonlinearities.rectify)
        print(l_avg_rr_s.output_shape)
        l_avg_rr_s = lasagne.layers.ReshapeLayer(l_avg_rr_s, (idxs_rr.shape[0],
                                                               max_post_length, rd))
        print(l_avg_rr_s.output_shape)
        '''
        
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        print(l_lstm_rr_s.output_shape)
        
        #LSTM w/ attn
        #now B x D
        custom_query = None
        d_attn = rd
        if d_title:
            custom_query = l_rr_title_s
            d_attn = d_title
                        
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], d_attn,
                                             custom_query=custom_query)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        #l_lstm_rr_avg = AverageSentenceLayer([l_lstm_rr_s, l_mask_rr_s])
        #l_lstm_rr_avg = lasagne.layers.SliceLayer(l_lstm_rr_s, indices=-1, axis=1)
        
        l_hid = l_lstm_rr_avg
        for num_layer in range(num_layers):
            l_hid = lasagne.layers.DenseLayer(l_hid, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)

            #now B x 1
            l_hid = lasagne.layers.DropoutLayer(l_hid, p_dropout)
        self.network = lasagne.layers.DenseLayer(l_hid, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)

        '''
        state = lasagne.layers.get_output(l_hid1)
        pos = state[::2, :]
        neg = state[1::2, :]
        weights = theano.shared(name='weights',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (rd,))).astype(theano.config.floatX)
        loss = margin_loss(weights, pos, neg) 
        predictions = T.nnet.sigmoid(weights[None,:] * state)
        params = lasagne.layers.get_all_params(l_hid1, trainable=True) + [weights]
        loss += lambda_w*T.sqrt(T.sum(weights**2)) + 0.*T.sum(gold)
        '''
        #TODO: dropout
        #now B x 1
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold, binary=True).mean()
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        #add regularization
        #mlp_params = [l_hid.W, l_hid.b, self.network.W, self.network.b]
        #loss += lambda_w*apply_penalty(mlp_params, l2)  #
        loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.adam(loss, params)
        #updates = lasagne.updates.adagrad(loss, params, learning_rate)
        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=learning_rate, momentum=0.9)

        print('compiling...')
        inputs = [idxs_rr, mask_rr_w, mask_rr_s]
        if d_title:
            inputs += [idxs_rr_title, mask_rr_title]
        self.train = theano.function(inputs + [gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True)
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        self.predict = theano.function(inputs,
                                       test_predictions, allow_input_downcast=True)

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        if not d_title:
            test_loss = lasagne.objectives.binary_hinge_loss(test_predictions, gold, binary=True).mean()        
            self.validate = theano.function([idxs_rr,
                                             mask_rr_w,
                                             mask_rr_s,
                                             gold, lambda_w, p_dropout],
                                            [loss, test_acc])

            #attention for words, B x S x N
            word_attention = lasagne.layers.get_output(AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d,
                                                                          W_w = l_attn_rr_w.W_w,
                                                                          u_w = l_attn_rr_w.u_w,
                                                                          b_w = l_attn_rr_w.b_w,
                                                                          normalized=False))
            self.word_attention = theano.function([idxs_rr,
                                                   mask_rr_w],
                                                   word_attention, allow_input_downcast=True)

            #attention for sentences, B x S
            sentence_attention = lasagne.layers.get_output(l_attn_rr_s)
            self.sentence_attention = theano.function([idxs_rr,
                                                    mask_rr_w,
                                                    mask_rr_s],
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
        
