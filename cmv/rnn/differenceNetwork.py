import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, BroadcastLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer, LSTMDecoderLayer

from theano.compile.nanguardmode import NanGuardMode

class DifferenceNetwork:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings,
                 GRAD_CLIP=100,
                 freeze_words=False,
                 shared=True):

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

        #just do an average over the words in both posts and take the difference
        #now B x S x N x D
        l_emb_op_w = lasagne.layers.EmbeddingLayer(l_idxs_op, V, d,
                                                   W=lasagne.utils.floatX(embeddings))
        if shared:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d, W=l_emb_op_w.W)
        else:
            l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                       W=lasagne.utils.floatX(embeddings))

        #reshape
        l_reshape_op = lasagne.layers.ReshapeLayer(l_emb_op_w,
                                                   (idxs_op.shape[0],
                                                    max_post_length*max_sentence_length, d))
        l_reshape_rr = lasagne.layers.ReshapeLayer(l_emb_rr_w,
                                                   (idxs_rr.shape[0],
                                                    max_post_length*max_sentence_length, d))
        l_reshape_op_mask = lasagne.layers.ReshapeLayer(l_mask_op_w,
                                                   (idxs_op.shape[0],
                                                    max_post_length*max_sentence_length))
        l_reshape_rr_mask = lasagne.layers.ReshapeLayer(l_mask_rr_w,
                                                   (idxs_rr.shape[0],
                                                    max_post_length*max_sentence_length))

        print(l_reshape_op.output_shape, l_reshape_rr.output_shape,
              l_reshape_op_mask.output_shape, l_reshape_rr_mask.output_shape)
        #average
        l_op_avg = AverageSentenceLayer([l_reshape_op, l_reshape_op_mask])
        l_rr_avg = AverageSentenceLayer([l_reshape_rr, l_reshape_rr_mask])
        #l_attn_rr_s = AttentionSentenceLayer([l_reshape_rr, l_reshape_rr_mask], rd)        
        #l_rr_avg = WeightedAverageSentenceLayer([l_reshape_rr, l_attn_rr_s])
        #l_attn_op_s = AttentionSentenceLayer([l_reshape_op, l_reshape_op_mask], rd)        
        #l_op_avg = WeightedAverageSentenceLayer([l_reshape_op, l_attn_op_s])
        print(l_op_avg.output_shape, l_rr_avg.output_shape)
        
        #rr-op
        l_diff = lasagne.layers.ElemwiseMergeLayer([l_rr_avg, l_op_avg],
                                                   T.sub)
        #l_diff = l_rr_avg #lasagne.layers.ConcatLayer([l_rr_avg, l_op_avg])
    
        #CBOW w/attn
        #now B x S x D
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, rd,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=GRAD_CLIP,
                                               mask_input=l_mask_rr_s)
        #LSTM w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_rr_s], rd)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])

        l_diff = lasagne.layers.ConcatLayer([l_diff, l_lstm_rr_avg])
        
                                                   
        l_hid1 = lasagne.layers.DenseLayer(l_diff, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)
        l_hid1 = lasagne.layers.DropoutLayer(l_hid1, p_dropout)
        l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)
        l_hid2 = lasagne.layers.DropoutLayer(l_hid2, p_dropout)
        self.network = lasagne.layers.DenseLayer(l_hid2, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()

        loss = lasagne.objectives.binary_crossentropy(T.clip(predictions,
                                                             1e-7,
                                                             1-(1e-7)),
                                                             gold).mean()
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold).mean()
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #add regularization
        lambda_w = T.scalar('lambda_w')
        loss += lambda_w*apply_penalty(params, l2)

        updates = lasagne.updates.adam(loss, params)
        #updates = lasagne.updates.nesterov_momentum(loss, params,
        #                                            learning_rate=0.01, momentum=0.9)

        print('compiling...')
        self.train = theano.function([idxs_op, idxs_rr,
                                      mask_op_w, mask_rr_w,
                                      mask_op_s, mask_rr_s,
                                      gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True,
                                     on_unused_input='warn',
                                     )#mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        self.predict = theano.function([idxs_op, idxs_rr,
                                        mask_op_w, mask_rr_w,
                                        mask_op_s, mask_rr_s,
                                        ],
                                       test_predictions, allow_input_downcast=True,
                                        on_unused_input='warn')

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        test_loss = lasagne.objectives.binary_crossentropy(test_predictions, gold).mean()        
        self.validate = theano.function([idxs_op, idxs_rr,
                                         mask_op_w, mask_rr_w,
                                         mask_op_s, mask_rr_s,
                                         gold, lambda_w, p_dropout],
                                        [loss, test_acc],
                                     on_unused_input='warn')

    def save(self, filename):
        param_values = lasagne.layers.get_all_param_values(self.network)
        np.savez_compressed(filename, *param_values)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
