import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.regularization import apply_penalty, l2

from cmv.rnn.preprocessing import MAX_POST_LENGTH, MAX_SENTENCE_LENGTH

class ArgumentationRNN:
    def __init__(self,
                 V,
                 d=100,
                 embeddings=None,
                 GRAD_CLIP=100):

        We = theano.shared(name='embeddings',
                           value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                           ).astype(theano.config.floatX)
        Wrr = theano.shared(name='Wrr',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                            ).astype(theano.config.floatX)
        Wh = theano.shared(name='Wh',
                           value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                           ).astype(theano.config.floatX)
        w = theano.shared(name='w',
                          value=0.2 * np.random.uniform(-1.0, 1.0, (d,))
                          ).astype(theano.config.floatX)

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_op = T.itensor3('idxs_op') #imatrix
        idxs_rr = T.itensor3('idxs_rr') #imatrix

        #now a B x S x D tensor
        x_op = We[idxs_op].mean(axis=2)
        x_rr = We[idxs_rr].mean(axis=2)

        #B x S matrix
        mask_op = T.imatrix('mask_op')
        mask_rr = T.imatrix('mask_rr')

        #now use this as an input to an LSTM
        l_in = lasagne.layers.InputLayer(shape=(None, MAX_POST_LENGTH, d) )#, input_var=x)
        l_mask = lasagne.layers.InputLayer(shape=(None, MAX_POST_LENGTH))

        #TODO: dropout

        #shape is still B x S x D
        lstm = lasagne.layers.LSTMLayer(l_in, d,
                                        nonlinearity=lasagne.nonlinearities.tanh,
                                        grad_clipping=GRAD_CLIP,
                                        mask_input=l_mask)

        #now B x D
        lstm_op = lstm.get_output_for([x_op, mask_op])[:, -1, :]
        lstm_rr = lstm.get_output_for([x_rr, mask_rr])[:, -1, :]

        #hidden state, still B x D
        f = T.tanh
        h = f(T.dot(lstm_op, Wh) + T.dot(lstm_rr, Wrr))

        #now see if that matches up with lstm_op, now a B-long vector
        predictions = T.nnet.sigmoid(T.batched_dot(-lstm_op, h))
        #predictions = T.nnet.sigmoid(T.dot(h, w))

        #also account for the fact that the root reply takes the opposite position
        #make this a constraint? minimize negative of lstm_op and lstm_rr
        #rr_predictions = T.nnet.sigmoid(T.batched_dot(-lstm_op, h))

        gold = T.ivector('gold')

        loss = lasagne.objectives.binary_crossentropy(predictions, gold).mean()

        params = lasagne.layers.get_all_params(lstm, trainable=True) + [We, Wrr, Wh]

        #add regularization
        lambda_w = T.scalar('lambda_w')
        loss += lambda_w*apply_penalty(params, l2)

        #could also add constraints that h initially is not convinced or that the opposing post is arguing the opposite side
        lambda_c = T.scalar('lambda_c')
        #minimize similarity between OP and root reply
        loss += lambda_c*T.sum(T.batched_dot(lstm_op, lstm_rr))
            
        updates = lasagne.updates.nesterov_momentum(
                    loss, params, learning_rate=0.01, momentum=0.9)

        self.train = theano.function([idxs_op, idxs_rr, mask_op, mask_rr, gold, lambda_w, lambda_c],
                                     loss, updates=updates, allow_input_downcast=True)

        self.predict = theano.function([idxs_op, idxs_rr, mask_op, mask_rr],
                                       predictions, allow_input_downcast=True)

        test_acc = T.mean(T.eq(predictions > .5, gold),
                                            dtype=theano.config.floatX)

        self.validate = theano.function([idxs_op, idxs_rr, mask_op, mask_rr, gold, lambda_w, lambda_c],
                                        [loss, test_acc])
        
