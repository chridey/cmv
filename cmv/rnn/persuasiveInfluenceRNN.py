import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer, MemoryLayer, AverageWordLayer, AverageSentenceLayer, MyConcatLayer

class PersuasiveInfluenceRNN:
    def __init__(self,
                 V,
                 d,
                 max_post_length,
                 max_sentence_length,
                 max_title_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 num_layers=1,
                 learning_rate=0.01,
                 add_biases=False,
                 rd=100,
                 op=False,
                 hops=3):

        print(V,d,max_post_length,max_sentence_length,max_title_length)

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_rr = T.itensor3('idxs_rr')
        idxs_op = T.itensor3('idxs_op')
        #B x T
        idxs_title = T.imatrix('idxs_title')
        mask_title = T.imatrix('mask_title')        
        #B x S x N matrix
        mask_rr_w = T.itensor3('mask_rr_w')
        mask_op_w = T.itensor3('mask_rr_w')
        #B x S matrix
        mask_rr_s = T.imatrix('mask_rr_s')
        mask_op_s = T.imatrix('mask_rr_s')        
        #B-long vector
        gold = T.ivector('gold')
        lambda_w = T.scalar('lambda_w')
        p_dropout = T.scalar('p_dropout')

        biases = T.matrix('biases')
        weights = T.ivector('weights')

        inputs = [idxs_rr, mask_rr_w, mask_rr_s] #CHANGE# ,
        if op:
            inputs.extend([idxs_op, mask_op_w, mask_op_s])
        
        #now use this as an input to an LSTM
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)
        l_idxs_op = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_op)
        l_mask_op_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_op_w)
        l_mask_op_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_op_s)
        l_idxs_title = lasagne.layers.InputLayer(shape=(None, max_title_length),
                                                 input_var=idxs_title)
        l_mask_title = lasagne.layers.InputLayer(shape=(None, max_title_length),
                                                 input_var=mask_title)
        
        if add_biases:
            l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                  input_var=biases)
        #now B x S x N x D
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        #now B x S x D
        #CHANGE
        #l_avg_rr_s = AverageWordLayer([l_emb_rr_w, l_mask_rr_w])
        
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        #CHANGE
        if False:
            l_avg_rr_s = HighwayLayer(l_avg_rr_s, num_units=l_avg_rr_s.output_shape[-1],
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    num_leading_axes=2)
               
        #separate embeddings for now (TODO)

        l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d,
                                                   W=lasagne.utils.floatX(embeddings))        
        l_attn_op_w = AttentionWordLayer([l_emb_op_w, l_mask_op_w], d)
        l_avg_op_s = WeightedAverageWordLayer([l_emb_op_w, l_attn_op_w])
        #bidirectional LSTM
        l_lstm_op_s_fwd = lasagne.layers.LSTMLayer(l_avg_op_s, rd,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=GRAD_CLIP,
                                                   mask_input=l_mask_op_s)
        l_lstm_op_s_rev = lasagne.layers.LSTMLayer(l_avg_op_s, rd,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=GRAD_CLIP,
                                                   mask_input=l_mask_op_s,
                                                   backwards=True)
        l_avg_op_s = lasagne.layers.ConcatLayer([l_lstm_op_s_fwd, l_lstm_op_s_rev], axis=-1)
        l_attn_op_s = AttentionSentenceLayer([l_avg_op_s, l_mask_op_s], d)
        l_op_avg = WeightedAverageSentenceLayer([l_avg_op_s, l_attn_op_s])
        
        '''
        l_avg_op_s = AverageWordLayer([l_emb_op_w, l_mask_op_w])
        l_emb_title = lasagne.layers.EmbeddingLayer(l_idxs_title, V, d,
                                                      W=l_emb_op_w.W)
        l_avg_title = AverageWordLayer([l_emb_title, l_mask_title])
        '''
            
        #CHANGE (LSTM here)
        #bidirectional LSTM
        l_lstm_rr_s_fwd = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=GRAD_CLIP,
                                                   mask_input=l_mask_rr_s)
        l_lstm_rr_s_rev = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=GRAD_CLIP,
                                                   mask_input=l_mask_rr_s,
                                                   backwards=True)

        #for first/last
        #l_lstm_rr_s_fwd_slice = lasagne.layers.SliceLayer(l_lstm_rr_s_fwd, indices=-1, axis=1)
        #l_lstm_rr_s_rev_slice = lasagne.layers.SliceLayer(l_lstm_rr_s_rev, indices=0, axis=1)
        #l_lstm_rr_s_bi = lasagne.layers.ConcatLayer([l_lstm_rr_s_fwd_slice, l_lstm_rr_s_rev_slice], axis=-1)
        #l_avg_rr_s = l_lstm_rr_s_bi

        #for attention or avergae
        l_avg_rr_s = lasagne.layers.ConcatLayer([l_lstm_rr_s_fwd, l_lstm_rr_s_rev], axis=-1)
        
        #CHANGE    
        #now memory network
        init_memory_response = AverageSentenceLayer([l_avg_rr_s, l_mask_rr_s])
        if op:
            init_memory_response = lasagne.layers.ConcatLayer([init_memory_response, l_op_avg])
        #init_memory_response_reshaped = lasagne.layers.DimshuffleLayer(init_memory_response,
        #                                                                (0, 'x', 1))
        l_memory = MyConcatLayer([l_avg_rr_s, init_memory_response])
                                                       
        #l_memory = lasagne.layers.ConcatLayer([l_avg_rr_s, init_memory_response_reshaped], axis=-1)
        #l_memory = lasagne.layers.ReshapeLayer(l_memory, ([0], max_post_length, [2]))
        l_attn_rr_s = AttentionSentenceLayer([l_avg_rr_s, l_mask_rr_s], d)
        l_rr_avg = WeightedAverageSentenceLayer([l_avg_rr_s, l_attn_rr_s])
        
        for i in range(hops):
            l_attn_rr_s = AttentionSentenceLayer([l_memory, l_mask_rr_s], d)
            l_rr_avg = WeightedAverageSentenceLayer([l_memory, l_attn_rr_s])
            if op:
                l_rr_avg = lasagne.layers.ConcatLayer([l_rr_avg, l_op_avg])
            l_memory = MyConcatLayer([l_avg_rr_s, l_rr_avg])
        #controller_state = MemoryLayer([l_avg_op_s, l_mask_op_s]) #TODO , query=l_avg_title) #, hops=d)
        #l_attn_rr_s = AttentionSentenceLayer([l_avg_rr_s, l_mask_rr_s], d, custom_query=controller_state,
        #                                     hidden_layers=0)
        #l_rr_avg = AverageSentenceLayer([l_avg_rr_s, l_mask_rr_s])
        
        l_hid = l_rr_avg
            
        for num_layer in range(num_layers):
            l_hid = lasagne.layers.DenseLayer(l_hid, num_units=d,
                                          nonlinearity=lasagne.nonlinearities.rectify)

            #now B x 1
            l_hid = lasagne.layers.DropoutLayer(l_hid, p_dropout)
            
        if add_biases:
            l_hid = lasagne.layers.ConcatLayer([l_hid, l_biases], axis=-1)
            inputs.append(biases)
            
        self.network = lasagne.layers.DenseLayer(l_hid, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        xent = lasagne.objectives.binary_crossentropy(predictions, gold)
        loss = lasagne.objectives.aggregate(xent, weights, mode='normalized_sum')
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        #add regularization
        loss += lambda_w*apply_penalty(params, l2)

        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=learning_rate, momentum=0.9)

        print('compiling...')
        train_outputs = loss
        self.train = theano.function(inputs + [gold, lambda_w, p_dropout, weights],
                                     train_outputs,
                                      updates=updates,
                                      allow_input_downcast=True,
                                      on_unused_input='warn')
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        
        self.predict = theano.function(inputs,
                                       test_predictions,
                                       allow_input_downcast=True,
                                      on_unused_input='warn')

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        test_loss = lasagne.objectives.binary_crossentropy(test_predictions,
                                                            gold).mean()        
        self.validate = theano.function(inputs + [gold, lambda_w, p_dropout, weights],
                                        [loss, test_acc],
                                      on_unused_input='warn')

        print('...')
        '''
        #attention for words, B x S x N        
        word_attention = lasagne.layers.get_output(AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d,
                                                                      W_w = l_attn_rr_w.W_w,
                                                                      u_w = l_attn_rr_w.u_w,
                                                                      b_w = l_attn_rr_w.b_w,
                                                                      normalized=False))
        self.word_attention = theano.function([idxs_rr,
                                               mask_rr_w],
                                               word_attention,
                                               allow_input_downcast=True,
                                               on_unused_input='warn')

        if d_frames:        
            frames_attention = lasagne.layers.get_output(AttentionWordLayer([l_emb_frames_rr_w, l_mask_rr_w], d,
                                                                          W_w = l_attn_rr_frames.W_w,
                                                                          u_w = l_attn_rr_frames.u_w,
                                                                          b_w = l_attn_rr_frames.b_w,
                                                                          normalized=False))
            self.frames_attention = theano.function([idxs_frames_rr,
                                                   mask_rr_w],
                                                   frames_attention,
                                                   allow_input_downcast=True,
                                                   on_unused_input='warn')
        '''
        #attention for sentences, B x S
        print('attention...')
        sentence_attention = lasagne.layers.get_output(l_attn_rr_s)
        if add_biases:
            inputs = inputs[:-1]
        self.sentence_attention = theano.function(inputs,
                                                  sentence_attention,
                                                  allow_input_downcast=True,
                                                  on_unused_input='warn')
        print('finished compiling...')
        
    def get_params(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params)
        
    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)
        
def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
