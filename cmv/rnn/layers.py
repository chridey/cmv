import theano
import theano.tensor as T
import numpy as np
import lasagne

#averaging layer
#bidirectional layer
#attention layer

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
                 u_w=lasagne.init.Normal(), b_w=lasagne.init.Normal(), **kwargs):
        super(AttentionWordLayer, self).__init__(incomings, **kwargs)
        self.W_w = self.add_param(W_w, (incomings[0].output_shape[-1],d))
        self.u_w = self.add_param(u_w, (d,))
        self.b_w = self.add_param(b_w, (1,))
        
    def get_output_for(self, inputs, **kwargs):
        #u = T.sum(inputs[0], axis=-1)
        u = T.dot(T.tanh(T.dot(inputs[0], self.W_w)), self.u_w)

        # set masked positions to large negative value
        u = u*inputs[1] - (1-inputs[1])*10000
        
        #now batch_size x post_length x sentence_length x 1 but need to normalize via softmax
        #over 2nd axis, and also multiply by the sentence mask

        # normalize over sentence_length (->large negative values = 0)
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
                 u_s=lasagne.init.Normal(), b_s=lasagne.init.Normal(), **kwargs):
        super(AttentionSentenceLayer, self).__init__(incomings, **kwargs)
        self.W_s = self.add_param(W_s, (incomings[0].output_shape[-1], d))
        self.u_s = self.add_param(u_s, (d,))
        self.b_s = self.add_param(b_s, (1,))
        
    def get_output_for(self, inputs, **kwargs):
        #u = T.sum(inputs[0], axis=-1)
        u = T.dot(T.tanh(T.dot(inputs[0], self.W_s)), self.u_s)

        # set masked positions to large negative value
        if len(inputs) > 1:
            u = u*inputs[1] - (1-inputs[1])*10000
        
        #now batch_size x post_length x 1 but need to normalize via softmax

        # normalize over post_length (->large negative values = 0)
        u = T.reshape(u, (inputs[0].shape[0], inputs[0].shape[1]))
        alpha = T.nnet.softmax(u)

        #now return the weighted sum
        #return T.sum(inputs[0] * alpha[:,:, None], axis=1)

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
        
#incomplete, only works for very specific case    
class BroadcastLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(BroadcastLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return T.ones_like(inputs[1]) * inputs[0][:, None, :]

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[1]
