import theano
import theano.tensor as T
import numpy as np
import lasagne

# compute vector average
class AverageWordLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(AverageWordLayer, self).__init__(incomings, **kwargs)

    #embedding layer is batch_size x max_post_length x max_sentence_length x  d
    #mask layer is batch_size x max_post_length x max_sentence_length 
    def get_output_for(self, inputs, **kwargs):
        emb_sums = T.sum(inputs[0] * inputs[1][:, :, :, None], axis=2)

        mask_sums = T.sum(inputs[1], axis=2)

        #need to avoid dividing by zero
        mask_sums += T.eq(mask_sums, T.as_tensor_variable(0))
        
        return emb_sums / mask_sums[:,:,None]

    # output is batch_size x max_post_length x d
    def get_output_shape_for(self, input_shapes):
        
        return (None,input_shapes[0][1],input_shapes[0][-1])

class AverageSentenceLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(AverageSentenceLayer, self).__init__(incomings, **kwargs)

    #sentence layer is batch_size x max_post_length x  d
    #mask layer is batch_size x max_post_length 
    def get_output_for(self, inputs, **kwargs):
        emb_sums = T.sum(inputs[0] * inputs[1][:, :, None], axis=1)
        mask_sums = T.sum(inputs[1], axis=1)

        return emb_sums / mask_sums[:,None]

    # output is batch_size x d
    def get_output_shape_for(self, input_shapes):
        
        return (None,input_shapes[0][-1])

class AttentionWordLayer(lasagne.layers.MergeLayer):
    #uses either a fixed "query" for the important words or another layer
    #this returns weights that can be used in the averaging layer in place of the mask
    def __init__(self, incomings, d, W_w=lasagne.init.Normal(),
                 u_w=lasagne.init.Normal(), b_w=lasagne.init.Normal(),
                 custom_query=None, normalized=True, **kwargs):
        super(AttentionWordLayer, self).__init__(incomings, **kwargs)
        self.W_w = self.add_param(W_w, (incomings[0].output_shape[-1],d))
        self.b_w = self.add_param(b_w, (d,))
        self.normalized = normalized

        self.fixed_query = True
        if custom_query is not None:
            self.fixed_query = False
            self.u_w = lasagne.layers.get_output(custom_query)
        else:
            self.u_w = self.add_param(u_w, (d,))        
        
    def get_output_for(self, inputs, **kwargs):
        #u = T.sum(inputs[0], axis=-1)
        if self.fixed_query:
            u = T.dot(T.tanh(T.dot(inputs[0], self.W_w) + self.b_w), self.u_w)
        else:
            u = T.batched_dot(T.tanh(T.dot(inputs[0], self.W_w) + self.b_w), self.u_w)
            
        # set masked positions to large negative value
        u = u*inputs[1] - (1-inputs[1])*10000
        
        #now batch_size x post_length x sentence_length x 1 but need to normalize via softmax
        #over 2nd axis, and also multiply by the sentence mask

        # normalize over sentence_length (->large negative values = 0)
        if not self.normalized:
            return T.reshape(u, (inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]))
        u = T.reshape(u, (inputs[0].shape[0]*inputs[0].shape[1], inputs[0].shape[2]))
        alpha = T.nnet.softmax(u)
        alpha = T.reshape(alpha, (inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]))

        #now return the weighted sum
        #return T.sum(inputs[0] * alpha[:,:,:, None], axis=2)

        return alpha
        
    def get_output_shape_for(self, input_shapes):
        
        #return (None,input_shapes[0][1],input_shapes[0][-1])
        return (None,input_shapes[0][1],input_shapes[0][2])

class AttentionSentenceLayer(lasagne.layers.MergeLayer):
    #uses either a fixed "query" for the important words or another layer
    #this returns weights that can be used in the averaging layer in place of the mask
    def __init__(self, incomings, d, W_s=lasagne.init.Normal(),
                 u_s=lasagne.init.Normal(), b_s=lasagne.init.Normal(),
                 custom_query=None, nonlinearity=T.tanh,
                 hidden_layers=1, **kwargs):
        super(AttentionSentenceLayer, self).__init__(incomings, **kwargs)
        self.W_s = [self.add_param(W_s, (incomings[0].output_shape[-1], d)) for i in range(hidden_layers)]
        self.b_s = [self.add_param(b_s, (d,)) for i in range(hidden_layers)]
        
        self.fixed_query = True
        if custom_query is not None:
            self.fixed_query = False
            self.u_s = lasagne.layers.get_output(custom_query)
        else:
            self.u_s = self.add_param(u_s, (d,))        
        self.nonlinearity = nonlinearity
        self.hidden_layers = hidden_layers
        
    def get_output_for(self, inputs, **kwargs):
        #u = T.sum(inputs[0], axis=-1)
        tmp = inputs[0]
        for i in range(self.hidden_layers):
            tmp = self.nonlinearity(T.dot(tmp, self.W_s[i]) + self.b_s[i][None, None, :])
                   
        if self.fixed_query:
            u = T.dot(tmp, self.u_s)            
        else:
            u = T.batched_dot(tmp, self.u_s)
        
        # set masked positions to large negative value
        if len(inputs) > 1:
            u = u*inputs[1] - (1-inputs[1])*10000
        
        #now batch_size x post_length x 1 but need to normalize via softmax

        # normalize over post_length (->large negative values = 0)
        u = T.reshape(u, (inputs[0].shape[0], inputs[0].shape[1]))
        alpha = T.nnet.softmax(u)

        #now return the weights

        return alpha
        
    def get_output_shape_for(self, input_shapes):
        
        #return (None,input_shapes[0][1],input_shapes[0][-1])
        return (None,input_shapes[0][-1])
    
class WeightedAverageWordLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(WeightedAverageWordLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1][:,:,:,None], axis=2)

    def get_output_shape_for(self, input_shapes):
        return (None, input_shapes[0][1], input_shapes[0][-1])

class WeightedAverageSentenceLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(WeightedAverageSentenceLayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1][:,:,None], axis=1)

    def get_output_shape_for(self, input_shapes):
        return (None, input_shapes[0][-1])
        
class HighwayLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W_h=lasagne.init.GlorotUniform(),
                 b_h=lasagne.init.Constant(0.), W_t=lasagne.init.GlorotUniform(),
                 b_t=lasagne.init.Constant(-2.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        
        super(HighwayLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_units = num_units

        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "leaving no trailing axes for the dot product." %
                (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                "Got num_leading_axes=%d for a %d-dimensional input, "
                "requesting more trailing axes than there are input "
                "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                "A DenseLayer requires a fixed input shape (except for "
                "the leading axes). Got %r for num_leading_axes=%d." %
                (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        assert(num_inputs == num_units)
        
        self.W_h = self.add_param(W_h, (num_inputs, num_units), name="W_h")
        if b_h is None:
            self.b_h = None
        else:
            self.b_h = self.add_param(b_h, (num_units,), name="b_h",
                                    regularizable=False)

        self.W_t = self.add_param(W_t, (num_inputs, num_units), name="W_t")
        if b_t is None:
            self.b_t = None
        else:
            self.b_t = self.add_param(b_t, (num_units,), name="b_t",
                                    regularizable=False)
            
    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        t = lasagne.nonlinearities.sigmoid(T.dot(input, self.W_t) + self.b_t)
        g = self.nonlinearity(T.dot(input, self.W_h) + self.b_h)

        return T.mul(t,g) + T.mul(1-t, input)

class MemoryLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, W_r=lasagne.init.GlorotUniform(),
                 hops=3, q=lasagne.init.Normal(), query=None, **kwargs):
        
        super(MemoryLayer, self).__init__(incomings, **kwargs)

        d = incomings[0].output_shape[-1]
        self.W_r = self.add_param(W_r, (d, d), name="W_r")
        self.hops = hops
        self.d = d
        
        self.fixed_query = True
        if query is not None:
            self.fixed_query = False
            self.q = lasagne.layers.get_output(query)
        else:
            self.q = self.add_param(q, (d,))        

    def get_output_shape_for(self, input_shape):
        #B x D
        return (None, self.d)
        
    def get_output_for(self, inputs, **kwargs):
        q = self.q
        for i in range(self.hops):
            if self.fixed_query and not i:
                u = T.dot(inputs[0], q)            
            else:
                u = T.batched_dot(inputs[0], q)

            # set masked positions to large negative value
            if len(inputs) > 1:
                u = u*inputs[1] - (1-inputs[1])*10000

            #now batch_size x post_length x 1 but need to normalize via softmax

            # normalize over post_length (->large negative values = 0)
            u = T.reshape(u, (inputs[0].shape[0], inputs[0].shape[1]))
            alpha = T.nnet.softmax(u)

            #now B x S
            o = T.dot(T.sum(inputs[0] * alpha[:,:,None], axis=1), self.W_r)
            if self.fixed_query:
                q = q + o
            else:
                q = q + o

        return q


class MyConcatLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(MyConcatLayer, self).__init__(incomings, **kwargs)  # MergeLayer constructor requires list of incoming layers
        
    def get_output_shape_for(self, input_shapes):
        lstm_shape, other_shape = input_shapes
        return (lstm_shape[0], lstm_shape[1], lstm_shape[2] + other_shape[-1])
    
    def get_output_for(self, inputs, **kwargs):
        lstm_input, other_input = inputs
        other_input = T.repeat(other_input.dimshuffle(0, 'x', 1), lstm_input.shape[1], axis=1)  # repeat along time dimension
        return T.concatenate((lstm_input, other_input), axis=-1)

