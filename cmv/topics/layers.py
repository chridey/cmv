import theano, cPickle, h5py, lasagne, random, csv, gzip                                                  
import numpy as np
import theano.tensor as T         


# rewritten embedding layer
class MyEmbeddingLayer(lasagne.layers.Layer):
    
    def __init__(self, incoming, input_size, output_size,
                 W=lasagne.init.Normal(), name='W', **kwargs):
        super(MyEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name=name)

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]

# compute vector average
class AverageLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, d, **kwargs):
        super(AverageLayer, self).__init__(incomings, **kwargs)
        self.d = d
        self.sum = True

    def get_output_for(self, inputs, **kwargs):
        emb_sums = T.sum(inputs[0] * inputs[1][:, :, None], axis=1)
        if self.sum:
            return emb_sums
        else:
            mask_sums = T.sum(inputs[1], axis=1)
            return emb_sums / mask_sums[:,None]

    # batch_size x max_spans x d
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0], self.d)

class AverageNegativeLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, d, **kwargs):
        super(AverageNegativeLayer, self).__init__(incomings, **kwargs)
        self.d = d
        self.sum = True

    def get_output_for(self, inputs, **kwargs):
        emb_sums = T.sum(inputs[0] * inputs[1][:, :, :, None], axis=2)
        if self.sum:
            return emb_sums
        else:
            mask_sums = T.sum(inputs[1], axis=2)
            return emb_sums / mask_sums[:,:,None]

    # batch_size x max_spans x d
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.d)
    

# multiply recurrent hidden states with descriptor matrix R
class ReconLayer(lasagne.layers.Layer):
    def __init__(self, incoming, d, num_descs, **kwargs):
        super(ReconLayer, self).__init__(incoming, **kwargs)
        self.R = self.add_param(lasagne.init.GlorotUniform(), 
            (num_descs, d), name='R')
        self.d = d
        
    def get_output_for(self, hiddens, **kwargs):
        return T.dot(hiddens, self.R)
        
    # batch_size x max_spans x d
    def get_output_shape_for(self, input_shapes):
        return (None, self.d)
