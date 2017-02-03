import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer
from cmv.rnn.loss import margin_loss

class ArgumentationMetadataRNN:
    def __init__(self,
                 sizes,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 frame_mask=False):

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_rr = T.itensor3('idxs_rr') 
        idxs_pos_rr = T.itensor3('idxs_pos_rr') 
        idxs_deps_rr = T.itensor3('idxs_deps_rr')
        idxs_govs_rr = T.itensor3('idxs_govs_rr')
        idxs_frames_rr = T.itensor3('idxs_frames_rr')
        idxs_clusters_rr = T.itensor3('idxs_clusters_rr')        
        
        #B x S x N matrix
        mask_rr_w = T.itensor3('mask_rr_w')
        #B x S matrix
        mask_rr_s = T.imatrix('mask_rr_s')

        #B-long vector
        gold = T.ivector('gold')
        lambda_w = T.scalar('lambda_w')
        p_dropout = T.scalar('p_dropout')
                        
        #now use this as an input to an LSTM
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_idxs_pos_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_pos_rr)
        l_idxs_deps_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_deps_rr)
        l_idxs_govs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_govs_rr)
        l_idxs_frames_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_frames_rr)
        l_idxs_clusters_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_clusters_rr)
        
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)

        masks = [mask_rr_w, mask_rr_s]
        if frame_mask:
            mask_rr_frames = T.itensor3('mask_rr_frames')
            l_mask_rr_frames = lasagne.layers.InputLayer(shape=(None, max_post_length,
                                                                max_sentence_length),
                                                         input_var=mask_rr_frames)
            masks += [mask_rr_frames]
            
        #now B x S x N x D
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr,
                                                   sizes['words']['V'],
                                                   sizes['words']['d'],
                                                   W=lasagne.utils.floatX(embeddings))

        #now concatenate all of these together
        inputs = [idxs_rr]
        if 'pos' in sizes:
            inputs += [idxs_pos_rr]
            l_emb_pos_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_pos_rr,
                                                            sizes['pos']['V'],
                                                            sizes['pos']['d'])
            l_emb_rr_w = lasagne.layers.ConcatLayer([l_emb_rr_w, l_emb_pos_rr_w], axis=-1)
        if 'deps' in sizes:
            inputs += [idxs_deps_rr]
            l_emb_deps_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_deps_rr,
                                                            sizes['deps']['V'],
                                                            sizes['deps']['d'])
            l_emb_rr_w = lasagne.layers.ConcatLayer([l_emb_rr_w, l_emb_deps_rr_w], axis=-1)
        if 'govs' in sizes:
            inputs += [idxs_govs_rr]
            l_emb_govs_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_govs_rr,
                                                            sizes['govs']['V'],
                                                            sizes['govs']['d'])

            l_emb_rr_w = lasagne.layers.ConcatLayer([l_emb_rr_w, l_emb_govs_rr_w], axis=-1)
        if 'frames' in sizes:
            print('adding frames...')
            inputs += [idxs_frames_rr]
            l_emb_frames_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_frames_rr,
                                                                sizes['frames']['V'],
                                                                sizes['frames']['d'])        
            l_emb_rr_w = lasagne.layers.ConcatLayer([l_emb_rr_w, l_emb_frames_rr_w], axis=-1)
        if 'clusters' in sizes:
            print('adding clusters...')
            inputs += [idxs_clusters_rr]
            l_emb_clusters_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_clusters_rr,
                                                                sizes['clusters']['V'],
                                                                sizes['clusters']['d'])        
            l_emb_rr_w = lasagne.layers.ConcatLayer([l_emb_rr_w, l_emb_clusters_rr_w], axis=-1)
            
        #CBOW w/attn
        #now B x S x D
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w],
                                         l_emb_rr_w.output_shape[-1])
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])
        
        '''        
        l_avg_rr_s_words = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        if frame_mask:
            l_attn_rr_frames = AttentionWordLayer([l_emb_frames_rr_w, l_mask_rr_frames],
                                                    l_emb_frames_rr_w.output_shape[-1])
        else:
            l_attn_rr_frames = AttentionWordLayer([l_emb_frames_rr_w, l_mask_rr_w],
                                                l_emb_frames_rr_w.output_shape[-1])
            
        l_avg_rr_s_frames = WeightedAverageWordLayer([l_emb_frames_rr_w, l_attn_rr_frames])

        l_attn_rr_pos = AttentionWordLayer([l_emb_pos_rr_w, l_mask_rr_w],
                                                l_emb_pos_rr_w.output_shape[-1])
            
        l_avg_rr_s_pos = WeightedAverageWordLayer([l_emb_pos_rr_w, l_attn_rr_pos])

        l_attn_rr_clusters = AttentionWordLayer([l_emb_clusters_rr_w, l_mask_rr_w],
                                                l_emb_clusters_rr_w.output_shape[-1])
            
        l_avg_rr_s_clusters = WeightedAverageWordLayer([l_emb_clusters_rr_w, l_attn_rr_clusters])

        
        
        l_avg_rr_s = lasagne.layers.ConcatLayer([l_avg_rr_s_words, l_avg_rr_s_frames, l_avg_rr_s_pos, l_avg_rr_s_clusters], axis=-1)

        '''
        
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        #LSTM w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])

        l_hid1 = lasagne.layers.DenseLayer(l_lstm_rr_avg, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)

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
        l_drop = lasagne.layers.DropoutLayer(l_hid1, p_dropout)
        self.network = lasagne.layers.DenseLayer(l_drop, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)        
        #now B x 1
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold, binary=True).mean()
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        #add regularization
        loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.adam(loss, params)
        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=0.01, momentum=0.9)

        print('compiling...')
        self.train = theano.function(inputs+masks+[gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True)
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        self.predict = theano.function(inputs+masks,
                                       test_predictions, allow_input_downcast=True)

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        #test_loss = lasagne.objectives.binary_hinge_loss(test_predictions, gold, binary=True).mean()
        test_loss = lasagne.objectives.binary_crossentropy(test_predictions, gold).mean()
                
        self.validate = theano.function(inputs+masks+[gold, lambda_w, p_dropout],
                                        [loss, test_acc])

        #attention for words, B x S x N
        #word_attention = lasagne.layers.get_output(l_attn_rr_w)
        #self.word_attention = theano.function(inputs[:1]+[mask_rr_w],
        #self.word_attention = theano.function(inputs+[mask_rr_w],
        #                                       word_attention, allow_input_downcast=True)
        #self.word_attention = theano.function(inputs+[mask_rr_w],
        #                                       word_attention, allow_input_downcast=True)
        #attention for frames, B x S x N
        '''
        frame_attention = lasagne.layers.get_output(l_attn_rr_frames)
        if frame_mask:
            self.frame_attention = theano.function(inputs[-1:]+[mask_rr_frames],
                                                frame_attention, allow_input_downcast=True)
        else:
            self.frame_attention = theano.function(inputs[-1:]+[mask_rr_w],
                                                frame_attention, allow_input_downcast=True)
        '''
        #attention for sentences, B x S
        sentence_attention = lasagne.layers.get_output(l_attn_rr_s)
        self.sentence_attention = theano.function(inputs+masks,
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
        
