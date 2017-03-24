import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer

class PersuasiveInfluenceRNN:
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
                 V_frames=0,
                 d_frames=0,
                 V_intra=0,
                 d_intra=0,
                 d_inter=0,
                 V_sentiment=0,
                 d_sentiment=0,
                 add_biases=False,
                 highway=True):

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_rr = T.itensor3('idxs_rr') #imatrix
        #B x S matrix of discourse tags
        idxs_disc_rr = T.imatrix('idxs_disc_rr')
        #B x S x N matrix
        mask_rr_w = T.itensor3('mask_rr_w')
        #B x S matrix
        mask_rr_s = T.imatrix('mask_rr_s')
        #B-long vector
        gold = T.ivector('gold')
        lambda_w = T.scalar('lambda_w')
        p_dropout = T.scalar('p_dropout')

        idxs_frames_rr = T.itensor3('idxs_frames_rr')
        #idxs_intra_rr = T.itensor3('idxs_intra_rr')
        idxs_intra_rr = T.imatrix('idxs_intra_rr')
        idxs_sentiment_rr = T.imatrix('idxs_sentiment_rr')

        biases = T.matrix('biases')
        weights = T.ivector('weights')
        
        inputs = [idxs_rr, mask_rr_w, mask_rr_s]
                
        #now use this as an input to an LSTM
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)
        l_idxs_frames_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_frames_rr)
        l_disc_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                    input_var=idxs_disc_rr)
        l_idxs_intra_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                    input_var=idxs_intra_rr)
        l_idxs_sentiment_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                    input_var=idxs_sentiment_rr)

        if add_biases:
            l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                  input_var=biases)
        #now B x S x N x D
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        #CBOW w/attn
        #now B x S x D
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s_words = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])
        concats = [l_avg_rr_s_words]
            
        if d_frames:
            assert(V_frames > 0)
            inputs += [idxs_frames_rr]
            l_emb_frames_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_frames_rr,
                                                              V_frames,
                                                              d_frames)        
        
            l_attn_rr_frames = AttentionWordLayer([l_emb_frames_rr_w, l_mask_rr_w],
                                                    l_emb_frames_rr_w.output_shape[-1])
            
            l_avg_rr_s_frames = WeightedAverageWordLayer([l_emb_frames_rr_w, l_attn_rr_frames])
            concats += [l_avg_rr_s_frames]
            
        if d_intra:
            assert(V_inter > 0)
            inputs += [idxs_disc_rr, idxs_intra_rr]
            
            l_emb_inter_rr_w = lasagne.layers.EmbeddingLayer(l_disc_idxs_rr,
                                                              V_inter,
                                                              d_intra)        

            
            l_emb_intra_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_intra_rr,
                                                              V_intra,
                                                              d_intra,
                                                              W=l_emb_inter_rr_w.W)
            concats += [l_emb_inter_rr_w, l_emb_intra_rr_w]
            
        if d_sentiment:
            assert(V_sentiment > 0)

            inputs += [idxs_sentiment_rr]
            l_emb_sentiment_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_sentiment_rr,
                                                              V_sentiment,
                                                              d_sentiment)
            
            concats += [l_emb_sentiment_rr_w]
            
        l_avg_rr_s = lasagne.layers.ConcatLayer(concats, axis=-1)

        #add MLP
        if highway:
            l_avg_rr_s = HighwayLayer(l_avg_rr_s, num_units=l_avg_rr_s.output_shape[-1],
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      num_leading_axes=2)
            
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        
        #LSTM w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        l_hid = l_lstm_rr_avg
            
        for num_layer in range(num_layers):
            l_hid = lasagne.layers.DenseLayer(l_hid, num_units=rd,
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
            
        #attention for sentences, B x S
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
        
