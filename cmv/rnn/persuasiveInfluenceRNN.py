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
                 embeddings=None,
                 GRAD_CLIP=100,
                 num_layers=1,
                 learning_rate=0.01,
                 add_biases=False,
                 rd=100,
                 op=False,
                 word_attn=True,
                 sent_attn=True,
                 highway=False,
                 hops=3,
                 words=True,
                 frames=False,
                 discourse=False):

        self._hyper_params = dict(V=V, d=d, max_post_length=max_post_length,
                                  max_sentence_length=max_sentence_length,
                                  GRAD_CLIP=GRAD_CLIP, num_layers=num_layers,
                                  learning_rate=learning_rate, add_biases=add_biases,
                                  rd=rd, op=op, word_attn=word_attn, sent_attn=sent_attn,
                                  highway=highway, hops=hops)
        
        print(V,d,max_post_length,max_sentence_length)

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_rr = T.itensor3('idxs_rr')
        idxs_op = T.itensor3('idxs_op')
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
        
        if add_biases:
            l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                  input_var=biases)
        #now B x S x N x D
        if embeddings is not None:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                    W=lasagne.utils.floatX(embeddings))
        else:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d)
                        
        #now B x S x D
        if words:
            if word_attn:
                l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
                l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])
            else:
                l_avg_rr_s = AverageWordLayer([l_emb_rr_w, l_mask_rr_w])
            concats = [l_avg_rr_s]
            inputs = [idxs_rr, mask_rr_w, mask_rr_s]        
        else:
            concats = []
            inputs = [mask_rr_w, mask_rr_s]
            
        if frames:
            idxs_frames_rr = T.itensor3('idxs_frames_rr')
            inputs.append(idxs_frames_rr)            
            l_idxs_frames_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                         input_var=idxs_frames_rr)
            l_emb_frames_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_frames_rr, V, d,
                                                              W=l_emb_rr_w.W)
            if word_attn:
                l_attn_rr_frames = AttentionWordLayer([l_emb_frames_rr_w, l_mask_rr_w], d)
                l_avg_rr_s_frames = WeightedAverageWordLayer([l_emb_frames_rr_w, l_attn_rr_frames])
            else:
                l_avg_rr_s_frames = AverageWordLayer([l_emb_frames_rr_w, l_mask_rr_w])
            concats.append(l_avg_rr_s_frames)

        if discourse:
            idxs_disc_rr = T.imatrix('idxs_disc_rr')
            inputs.append(idxs_disc_rr)
            l_emb_disc_rr = lasagne.layers.EmbeddingLayer(l_idxs_disc_rr, V, d,
                                                          W=l_emb_rr_w.W)
            concats.append(l_emb_disc_rr)
            
        l_avg_rr_s = lasagne.layers.ConcatLayer(concats, axis=-1)
                
        if highway:
            l_avg_rr_s = HighwayLayer(l_avg_rr_s, num_units=l_avg_rr_s.output_shape[-1],
                                    nonlinearity=lasagne.nonlinearities.rectify,
                                    num_leading_axes=2)
               
        #separate embeddings for OP
        if embeddings is not None:
            l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d,
                                                    W=lasagne.utils.floatX(embeddings))
        else:
            l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d)

        if op:
            if words:            
                l_attn_op_w = AttentionWordLayer([l_emb_op_w, l_mask_op_w], d)
                l_avg_op_s = WeightedAverageWordLayer([l_emb_op_w, l_attn_op_w])
                concats = [l_avg_op_s]
                inputs.extend([idxs_op, mask_op_w, mask_op_s])
            else:
                concats = []
                inputs.extend([mask_op_w, mask_op_s])
                
            if frames:
                idxs_frames_op = T.itensor3('idxs_frames_op')
                inputs.append(idxs_frames_op)            
                l_idxs_frames_op = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                             input_var=idxs_frames_op)
                l_emb_frames_op_w = lasagne.layers.EmbeddingLayer(l_idxs_frames_op, V, d,
                                                                  W=l_emb_op_w.W)
                l_attn_op_frames = AttentionWordLayer([l_emb_frames_op_w, l_mask_op_w], d)
                l_avg_op_s_frames = WeightedAverageWordLayer([l_emb_frames_op_w, l_attn_op_frames])
                concats.append(l_avg_op_s_frames)

            if discourse:
                idxs_disc_op = T.imatrix('idxs_disc_op')
                inputs.append(idxs_disc_op)
                l_emb_disc_op = lasagne.layers.EmbeddingLayer(l_idxs_disc_op, V, d,
                                                              W=l_emb_op_w.W)
                concats.append(l_emb_disc_op)

            l_avg_op_s = lasagne.layers.ConcatLayer(concats, axis=-1)

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

        #for attention or avergae
        l_lstm_rr_s = lasagne.layers.ConcatLayer([l_lstm_rr_s_fwd, l_lstm_rr_s_rev], axis=-1)
        
        #now memory network
        init_memory_response = AverageSentenceLayer([l_lstm_rr_s, l_mask_rr_s])
        if op:
            init_memory_response = lasagne.layers.ConcatLayer([init_memory_response, l_op_avg])
        l_memory = MyConcatLayer([l_lstm_rr_s, init_memory_response])

        if sent_attn:                                                       
            l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], d)
            l_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        else:
            l_rr_avg = AverageSentenceLayer([l_lstm_rr_s, l_mask_rr_s])
            
        for i in range(hops):
            l_attn_rr_s = AttentionSentenceLayer([l_memory, l_mask_rr_s], d)
            l_rr_avg = WeightedAverageSentenceLayer([l_memory, l_attn_rr_s])
            if op:
                l_rr_avg = lasagne.layers.ConcatLayer([l_rr_avg, l_op_avg])
            l_memory = MyConcatLayer([l_lstm_rr_s, l_rr_avg])
        
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

        #attention for words, B x S x N
        print('attention...')        
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

        #attention for sentences, B x S
        print('...')
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

    @property
    def hyper_params(self):
        return self._hyper_params
                
    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)
        
def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
