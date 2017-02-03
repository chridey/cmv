import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer, LSTMDiscourseLayer
from cmv.rnn.loss import margin_loss

class ArgumentationDiscourseRNN:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 discourse_tagged=False,
                 discourse_predictions=False,
                 K=1,
                 num_layers=1,
                 learning_rate=0.01,
                 V_frames=0,
                 d_frames=0,
                 V_intra=0,
                 d_intra=0,
                 V_sentiment=0,
                 d_sentiment=0,
                 V_chars=0,
                 d_chars=0,
                 max_chars_length=0,
                 causal=False,
                 post_features_length=0,
                 add_biases=False,
                 num_filters=30,
                 filter_length_range=(1,5),
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
        lambda_k = T.scalar('lambda_w')        
        p_dropout = T.scalar('p_dropout')

        idxs_frames_rr = T.itensor3('idxs_frames_rr')
        #idxs_intra_rr = T.itensor3('idxs_intra_rr')
        idxs_intra_rr = T.imatrix('idxs_intra_rr')
        idxs_sentiment_rr = T.imatrix('idxs_sentiment_rr')

        idxs_causal_rr = T.itensor3('idxs_causal_rr')
        #idxs_causal_rr = T.tensor3('idxs_causal_rr')
        idxs_chars_rr = T.itensor3('idxs_chars_rr')
        
        post_features = T.matrix('post_features')
        biases = T.matrix('biases')
        weights = T.ivector('weights')
        
        inputs = [idxs_rr, mask_rr_w, mask_rr_s]
                
        #now use this as an input to an LSTM
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_disc_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                            input_var=idxs_disc_rr)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)
        l_idxs_frames_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_frames_rr)
        #l_idxs_intra_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
        #                                   input_var=idxs_intra_rr)
        l_idxs_intra_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                    input_var=idxs_intra_rr)
        l_idxs_sentiment_rr = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                    input_var=idxs_sentiment_rr)
        l_idxs_causal_rr = lasagne.layers.InputLayer(shape=(None, max_post_length,
                                                             max_sentence_length),
                                                             #1),
                                            input_var=idxs_causal_rr)

        if post_features_length:
            l_post_features = lasagne.layers.InputLayer(shape=(None, post_features_length),
                                                        input_var=post_features)
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

        if d_chars:
            l_idxs_chars_rr = lasagne.layers.InputLayer(shape=(None, max_post_length,
                                                             max_chars_length),
                                                input_var=idxs_chars_rr)
            
            inputs += [idxs_chars_rr]
            #CNN w/ max pooling
            #need to reshape input to be (batch_size*max_post_length, 1, max_chars_length, d)
            l_emb_rr_chars = lasagne.layers.EmbeddingLayer(l_idxs_chars_rr, V_chars, d_chars)
            l_reshape_emb_rr_chars = lasagne.layers.ReshapeLayer(l_emb_rr_chars,
                                                             (-1, 1, max_chars_length, d_chars))
            filter_concats = []
            print(l_emb_rr_chars.output_shape)
            print(l_reshape_emb_rr_chars.output_shape)
            for i in range(filter_length_range[0], filter_length_range[1]+1):
                print(i, max_chars_length-i+1)
                l_emb_conv = lasagne.layers.Conv2DLayer(l_reshape_emb_rr_chars,
                                                        num_filters=num_filters, filter_size=(i, d_chars),
                                                        nonlinearity=lasagne.nonlinearities.rectify,
                                                        W=lasagne.init.GlorotUniform())
                print(l_emb_conv.output_shape)
                l_pool = lasagne.layers.MaxPool2DLayer(l_emb_conv, pool_size=(max_chars_length-i+1,1))
                #now of shape (batch_size*max_post_length, num_filters, 1, 1)
                l_pool_reshape = lasagne.layers.ReshapeLayer(l_pool,
                                                             (-1, max_post_length, num_filters))
                filter_concats.append(l_pool_reshape)

            l_conv_rr_s_chars = lasagne.layers.ConcatLayer(filter_concats, axis=-1)
            concats += [l_conv_rr_s_chars]

        
        '''
            
        l_avg_rr_s = lasagne.layers.ConcatLayer(concats, axis=-1)            
        
        if discourse_predictions:
            l_avg_rr_s_fwd = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                                  nonlinearity=lasagne.nonlinearities.tanh,
                                                  grad_clipping=GRAD_CLIP,
                                                  mask_input=l_mask_rr_s)        
            l_avg_rr_s_rev = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                                  nonlinearity=lasagne.nonlinearities.tanh,
                                                  grad_clipping=GRAD_CLIP,
                                                  mask_input=l_mask_rr_s,
                                                  backwards=True)
            l_avg_rr_s = lasagne.layers.ConcatLayer([l_avg_rr_s_fwd, l_avg_rr_s_rev], axis=-1)

        lstm_type = LSTMDiscourseLayer
        if discourse_predictions:
            lstm_type = LSTMSoftDiscourseLayer
        lstm_incomings = l_avg_rr_s
        if discourse_tagged and not discourse_predictions:
            lstm_incomings = [l_avg_rr_s, l_disc_idxs_rr]
            
        l_lstm_rr_s = lstm_type(lstm_incomings, rd, K,
                                nonlinearity=lasagne.nonlinearities.tanh,
                                grad_clipping=GRAD_CLIP,
                                mask_input=l_mask_rr_s)
        '''
            
        if discourse_tagged:
            inputs.append(idxs_disc_rr)                            
            if discourse_predictions:
                inputs.append(lambda_k)
            
            assert(K > 0)
            print(K)
            assert(d_intra > 0)
            l_emb_inter_rr_w = lasagne.layers.EmbeddingLayer(l_disc_idxs_rr,
                                                              K,
                                                              d_intra)        
            concats += [l_emb_inter_rr_w]
            
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
            assert(V_intra > 0)
            print(V_intra)
            assert(discourse_tagged)
            inputs += [idxs_intra_rr]
            l_emb_intra_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_intra_rr,
                                                              V_intra,
                                                              d_intra,
                                                              W=l_emb_inter_rr_w.W)        
            concats += [l_emb_intra_rr_w]
        '''
             
        if d_intra:
            assert(V_intra > 0)
            print(V_intra)
            assert(discourse_tagged)
            
            inputs += [idxs_intra_rr]
            l_emb_intra_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_intra_rr,
                                                              V_intra,
                                                              d_intra,
                                                              W=l_emb_inter_rr_w.W)        
        
            l_attn_rr_intra = AttentionWordLayer([l_emb_intra_rr_w, l_mask_rr_w],
                                                    l_emb_intra_rr_w.output_shape[-1])
            
            l_avg_rr_s_intra = WeightedAverageWordLayer([l_emb_intra_rr_w, l_attn_rr_intra])
            concats += [l_avg_rr_s_intra]
        '''
            
        if d_sentiment:
            assert(V_sentiment > 0)
            print(V_sentiment)

            inputs += [idxs_sentiment_rr]
            l_emb_sentiment_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_sentiment_rr,
                                                              V_sentiment,
                                                              d_sentiment)
            
            concats += [l_emb_sentiment_rr_w]
            
        if causal:
            inputs += [idxs_causal_rr]
            l_attn_rr_causal = AttentionWordLayer([l_emb_rr_w, l_idxs_causal_rr], d)
            l_avg_rr_s_causal = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_causal])
            concats += [l_avg_rr_s_causal]
            #concats += [l_idxs_causal_rr]
            
        l_avg_rr_s = lasagne.layers.ConcatLayer(concats, axis=-1)
        #add MLP
        print(l_avg_rr_s.output_shape)
        #l_avg_rr_s = lasagne.layers.ReshapeLayer(l_avg_rr_s, (-1, l_avg_rr_s.output_shape[-1]))
        #print(l_avg_rr_s.output_shape)
        if highway:
            l_avg_rr_s = HighwayLayer(l_avg_rr_s, num_units=l_avg_rr_s.output_shape[-1],
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      num_leading_axes=2)
        print(l_avg_rr_s.output_shape)
        #l_avg_rr_s = lasagne.layers.ReshapeLayer(l_avg_rr_s, (-1, max_post_length,
        #                                                      l_avg_rr_s.output_shape[-1]))
        #print(l_avg_rr_s.output_shape)
        #TODO: add highway
        
        #l_avg_rr_meta = lasagne.layers.ConcatLayer(concats[1:], axis=-1)
        #l_avg_rr_s = concats[0]
        #TODO: separate LSTM for structure?
        #TODO: separate embeddings for intra and inter?
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        '''                                               
        l_lstm_rr_s_meta = lasagne.layers.LSTMLayer(l_avg_rr_meta, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        l_lstm_rr_s = lasagne.layers.ConcatLayer([l_lstm_rr_s, l_lstm_rr_s_meta], axis=-1)
                                               

        '''
        print(l_lstm_rr_s.output_shape)
        
        #LSTM w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        #l_lstm_rr_avg = AverageSentenceLayer([l_lstm_rr_s, l_mask_rr_s])
        #l_lstm_rr_avg = lasagne.layers.SliceLayer(l_lstm_rr_s, indices=-1, axis=1)

        l_hid = l_lstm_rr_avg
        if post_features_length:
            l_post_features_hid = lasagne.layers.DenseLayer(l_post_features,
                                                            num_units=post_features_length,
                                                            nonlinearity=lasagne.nonlinearities.tanh)
            l_hid = lasagne.layers.ConcatLayer([l_hid, l_post_features_hid], axis=-1)
            inputs.append(post_features)
            
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
        
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold, binary=True).mean()
        xent = lasagne.objectives.binary_crossentropy(predictions, gold)
        loss = lasagne.objectives.aggregate(xent, weights, mode='normalized_sum')
        #loss = xent.mean()
        
        #TODO: add loss function for discourse tags
        #predictions are B x S x K
        if discourse_tagged and discourse_predictions:
            pred_reshape = l_lstm_rr_s.predictions.reshape((-1,
                                                           K))
            disc_reshape = idxs_disc_rr.reshape((-1,))
            aux_loss = lasagne.objectives.categorical_crossentropy(pred_reshape,
                                                                   disc_reshape).mean()
            loss += lambda_k*aux_loss
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        #add regularization
        #mlp_params = [l_hid.W, l_hid.b, self.network.W, self.network.b]
        #loss += lambda_w*apply_penalty(mlp_params, l2)  #
        loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.adam(loss, params)
        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=learning_rate, momentum=0.9)

        print('compiling...')
        #TODO: add idxs_disc_rr if discourse_tagged
        #update loss function if discourse_tagged and using soft discourse

        train_outputs = loss
        if discourse_predictions:
            train_outputs = [loss, l_lstm_rr_s.class_counts]
            
        self.train = theano.function(inputs + [gold, lambda_w, p_dropout, weights],
                                     train_outputs,
                                      updates=updates,
                                      allow_input_downcast=True,
                                      on_unused_input='warn')
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        if discourse_predictions:
            test_predictions = [test_predictions, l_lstm_rr_s.class_counts]
        
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
        if post_features_length or add_biases:
            inputs = inputs[:-1]
        self.sentence_attention = theano.function(inputs,
                                                  sentence_attention,
                                                  allow_input_downcast=True,
                                                  on_unused_input='warn')
        print('finished compiling...')

    def save(self, filename):
        param_values = lasagne.layers.get_all_param_values(self.network)
        np.savez_compressed(filename, *param_values)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
