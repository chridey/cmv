import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.layers import AverageWordLayer, AverageSentenceLayer, AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer
from cmv.rnn.lstm_layers import reshapeLSTMLayer
from cmv.rnn.loss import margin_loss

class ArgumentationDAN:
    def __init__(self,
                 V,
                 d,
                 rd,
                 max_post_length,
                 max_sentence_length,
                 embeddings=None,
                 GRAD_CLIP=100,
                 weighted=True,
                 hierarchical=True,
                 num_layers=2,
                 freeze_words=False):

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
        
        #now use this as input
        l_idxs_rr = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                            input_var=idxs_rr)
        l_mask_rr_w = lasagne.layers.InputLayer(shape=(None, max_post_length, max_sentence_length),
                                                input_var=mask_rr_w)
        l_mask_rr_s = lasagne.layers.InputLayer(shape=(None, max_post_length),
                                                input_var=mask_rr_s)
        
        #now B x S x N x D
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_rr, V, d,
                                                   W=lasagne.utils.floatX(embeddings))

        #CBOW w/attn
        #now B x S x D
        l_attn_rr_w = AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d)
        l_avg_rr_s = WeightedAverageWordLayer([l_emb_rr_w, l_attn_rr_w])

        #add an MLP here??
        
        #CBOS w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_avg_rr_s, l_mask_rr_s], rd)        
        l_avg_rr_p = WeightedAverageSentenceLayer([l_avg_rr_s, l_attn_rr_s])

        #now B x RD
        l_hid1 = lasagne.layers.DenseLayer(l_avg_rr_p, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)

        l_drop1 = lasagne.layers.DropoutLayer(l_hid1, p_dropout)
        l_hid2 = lasagne.layers.DenseLayer(l_drop1, num_units=rd,
                                          nonlinearity=lasagne.nonlinearities.rectify)

        l_drop2 = lasagne.layers.DropoutLayer(l_hid2, p_dropout)
        #now B x 1        
        self.network = lasagne.layers.DenseLayer(l_drop2, num_units=1,
                                                 nonlinearity=T.nnet.sigmoid)
        
        predictions = lasagne.layers.get_output(self.network).ravel()
        
        #loss = lasagne.objectives.binary_hinge_loss(predictions, gold, binary=True).mean()
        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        #add regularization
        mlp_params = [l_hid1.W, l_hid1.b, self.network.W, self.network.b]
        #loss += lambda_w*apply_penalty(mlp_params, l2)  #
        loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.adam(loss, params)
        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=0.01, momentum=0.9)

        print('compiling...')
        self.train = theano.function([idxs_rr,
                                      mask_rr_w,
                                      mask_rr_s,
                                      gold, lambda_w, p_dropout],
                                     loss, updates=updates, allow_input_downcast=True)
        print('...')
        test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        self.predict = theano.function([idxs_rr,
                                        mask_rr_w,
                                        mask_rr_s,
                                        ],
                                       test_predictions, allow_input_downcast=True)

        test_acc = T.mean(T.eq(test_predictions > .5, gold),
                                            dtype=theano.config.floatX)
        print('...')
        test_loss = lasagne.objectives.binary_hinge_loss(test_predictions, gold, binary=True).mean()        
        self.validate = theano.function([idxs_rr,
                                         mask_rr_w,
                                         mask_rr_s,
                                         gold, lambda_w, p_dropout],
                                        [loss, test_acc])

        #attention for words, B x S x N
        word_attention = lasagne.layers.get_output(AttentionWordLayer([l_emb_rr_w, l_mask_rr_w], d,
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
        
