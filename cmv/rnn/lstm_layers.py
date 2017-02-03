import theano
import theano.tensor as T
import numpy as np
import lasagne

#averaging layer
#bidirectional layer
#attention layer
            
def reshapeLSTMLayer(embedding_layer, mask_layer, input_shape, d):
    l_reshape_op_w = lasagne.layers.ReshapeLayer(embedding_layer,
                                                (input_shape[0]*input_shape[1],
                                                input_shape[2], input_shape[3]))
    l_reshape_mask_op_w = lasagne.layers.ReshapeLayer(mask_layer,
                                                      (input_shape[0]*input_shape[1],
                                                       input_shape[2]))
    l_lstm_op_w = lasagne.layers.LSTMLayer(l_reshape_op_w, d,
                                           nonlinearity=lasagne.nonlinearities.tanh,
                                           grad_clipping=100,
                                           mask_input=l_reshape_mask_op_w)
    return lasagne.layers.ReshapeLayer(l_lstm_op_w,
                                       (input_shape[0], input_shape[1],
                                        input_shape[2], input_shape[3]))

class LSTMDiscourseLayer(lasagne.layers.LSTMLayer):
    def __init__(self, incomings, d, K, **kwargs):
        '''
        incomings is the previous layer plus the discourse tags
          previous layer is B x S x D
          discourse tags are B x S
        K is the number of possible connectives
        '''
        super(LSTMDiscourseLayer, self).__init__(incomings[0], d, **kwargs)
        self.K = K
        
        #initialize weights for discourse relation at time t
        self.W_hid_to_ingate = self.add_param(lasagne.init.Normal(), 
                                            (K,d,d), 
                                            name='W_hid_to_ingate')
        self.W_hid_to_forgetgate = self.add_param(lasagne.init.Normal(), 
                                            (K,d,d), 
                                            name='W_hid_to_forgetgate')
        self.W_hid_to_cell = self.add_param(lasagne.init.Normal(), 
                                            (K,d,d), 
                                            name='W_hid_to_cell')
        self.W_hid_to_outgate = self.add_param(lasagne.init.Normal(), 
                                            (K,d,d), 
                                            name='W_hid_to_outgate')
        
        self.input_shapes.append(incomings[1] if isinstance(incomings[1], 
                                                            tuple) else incomings[1].output_shape)
        self.input_layers.append(None if isinstance(incomings[1],
                                                            tuple) else incomings[1])

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        discourse = inputs[-1]
        
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        #same with discourse, now S x B
        discourse = discourse.dimshuffle(1,0)
        
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=-1)

        # Same for hidden weight matrices
        #now K x in_units x 4*num_units
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=-1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

            #need to calculate the weight over the discourse connective

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        # discourse_n is the index of the discourse relation at time n
        def step(input_n, discourse_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.batched_dot(hid_previous, W_hid_stacked[discourse_n])

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, discourse_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, discourse_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            
            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, discourse, mask]
            step_fun = step_masked
        else:
            sequences = [input, discourse]
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

    #this is the same as LSTMLayer
    #def get_output_shape_for(self, input_shapes):
    #    pass

class LSTMSoftDiscourseLayer(lasagne.layers.LSTMLayer):
    '''
    incomings - input layer (B x S x D) 
    '''
    def __init__(self, incomings, d, K, **kwargs):
        #K is the number of possible connectives
        super(LSTMSoftDiscourseLayer, self).__init__(incomings, d, **kwargs)
        self.K = K
        
        #initialize weights for discourse relation at time t
        num_inputs = incomings.output_shape[-1]

        self.W_hid_to_ingate = self.add_param(lasagne.init.Normal(), 
                                            (d,K*d), 
                                            name='W_hid_to_ingate')
        self.W_hid_to_forgetgate = self.add_param(lasagne.init.Normal(), 
                                            (d,K*d), 
                                            name='W_hid_to_forgetgate')
        self.W_hid_to_cell = self.add_param(lasagne.init.Normal(), 
                                            (d,K*d), 
                                            name='W_hid_to_cell')
        self.W_hid_to_outgate = self.add_param(lasagne.init.Normal(), 
                                            (d,K*d), 
                                            name='W_hid_to_outgate')
        
        self.W_disc = self.add_param(lasagne.init.Normal(),
                                     (d,K),
                                     name='W_disc')
        self.b_disc = self.add_param(lasagne.init.Normal(),
                                     (K,),
                                     name='b_disc')

        self.class_counts = None
                
    def prob_discourse(self, prev_hidden):
        #prev_hidden is batch_size * num_units
        #return T.ones((prev_hidden.shape[0], self.K))*1./self.K
        return T.nnet.softmax(T.dot(prev_hidden, self.W_disc) + self.b_disc[None,:])
        
    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]

        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=-1)

        # Same for hidden weight matrices
        #now in_units x 4*K*num_units
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=-1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

            #need to calculate the weight over the discourse connective

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, class_counts, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            #this is n_batch x K
            disc_prob = self.prob_discourse(hid_previous)
            #this is (n_batch, 4*K*num_units)            
            hidden_temp = T.dot(hid_previous, W_hid_stacked).reshape((input_n.shape[0],
                                                                     4, self.K, self.num_units))
            hidden_temp = (hidden_temp * disc_prob[:, None, :, None]).sum(axis=-2).reshape((input_n.shape[0], 4*self.num_units))
            
            # Calculate gates pre-activations and slice
            gates = input_n + hidden_temp

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid, class_counts+disc_prob]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, class_counts, *args):
            cell, hid, counts = step(input_n, cell_previous, hid_previous, class_counts, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            counts = T.switch(mask_n, counts, class_counts)
            
            return [cell, hid, counts]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        non_seqs += [self.W_disc, self.b_disc]
            
        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, counts_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, T.zeros((num_batch, self.K))],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, counts_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, T.zeros((num_batch, self.K))],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        self.class_counts = counts_out[-1]
                
        return hid_out

    #this is the same as LSTMLayer
    #def get_output_shape_for(self, input_shapes):
    #    pass
    
class LSTMDecoderLayer(lasagne.layers.LSTMLayer):
    '''
    incomings - input layer (B x S x D) and context layer (either B x S x D or B x D)
    '''
    def __init__(self, incomings, d, alignment=False, context_mask=None, **kwargs):
        super(LSTMDecoderLayer, self).__init__(incomings[0], d, **kwargs)

        #initialize weights for context at time t
        num_inputs = incomings[1].output_shape[-1]
        self.W_c_to_ingate = self.add_param(lasagne.init.Normal(), 
                                            (num_inputs,d), 
                                            name='W_c_to_ingate')
        self.W_c_to_forgetgate = self.add_param(lasagne.init.Normal(), 
                                            (num_inputs,d), 
                                            name='W_c_to_forgetgate')
        self.W_c_to_cell = self.add_param(lasagne.init.Normal(), 
                                            (num_inputs,d), 
                                            name='W_c_to_cell')
        self.W_c_to_outgate = self.add_param(lasagne.init.Normal(), 
                                            (num_inputs,d), 
                                            name='W_c_to_outgate')

        self.input_shapes.append(incomings[1] if isinstance(incomings[1], 
                                                            tuple) else incomings[1].output_shape)
        self.input_layers.append(None if isinstance(incomings[1],
                                                            tuple) else incomings[1])
                                 

        #initialize alignment layer (optional)
        #need to add context mask in this case
        self.alignment = alignment
        if alignment:
            pass #do stuff

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # retrieve the context
        context = inputs[-1]

        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Same for context weight matrices
        W_c_stacked = T.concatenate(
            [self.W_c_to_ingate, self.W_c_to_forgetgate,
             self.W_c_to_cell, self.W_c_to_outgate], axis=1)        

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        if not self.alignment: 
            #can precompute if context is independent of input at time t
            context = T.dot(context, W_c_stacked)

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked) + context

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if not self.alignment:
            non_seqs += [context]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

    #this is the same as LSTMLayer
    #def get_output_shape_for(self, input_shapes):
    #    pass
