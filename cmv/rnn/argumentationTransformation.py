import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, BroadcastLayer, CosineSimilarityLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer, LSTMDecoderLayer

#treat the original post (p) as a vector and the response (R) as a diagonal matrix
# so if R has no effect on p, pR will be close to p
#so maximize pRp for negative examples, minimize pRp for positive examples

#randomly sample some other examples to make them orthogonal?

class ArgumentationTransformation:
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

        print('using projection...')
        
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

        l_proj = lasagne.layers.ElemwiseMergeLayer([l_lstm_op_avg, l_lstm_rr_avg],
                                                   T.mul)
        #shape is B x D
        l_cos = CosineSimilarityLayer([l_proj, l_lstm_op_avg])
        #now B x 1
        
        self.network = lasagne.layers.DenseLayer(l_cos, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
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
                                      gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True,
                                     on_unused_input='warn')
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
        
