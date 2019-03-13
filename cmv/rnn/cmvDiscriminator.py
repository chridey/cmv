from typing import Sequence, Union, Dict

import torch

from allennlp.models import Model

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.modules import FeedForward

from allennlp.training.metrics import BooleanAccuracy

@Model.register("cmv_discriminator")
class CMVDiscriminator(FeedForward):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, Sequence[int]],
                 activations: Union[Activation, Sequence[Activation]],
                 dropout: Union[float, Sequence[float]] = 0.0,
                 gate_bias: float = -2) -> None:

        super(CMVDiscriminator, self).__init__(input_dim, num_layers, hidden_dims,
                                              activations, dropout)

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * (num_layers-1)
        input_dims = hidden_dims[1:]
        
        gate_layers = [None] #so we can zip this later
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            gate_layer = torch.nn.Linear(layer_input_dim, layer_output_dim)
            gate_layer.bias.data.fill_(gate_bias)

            gate_layers.append(gate_layer)
            
        self._gate_layers = torch.nn.ModuleList(gate_layers)

        #feedforward requires an Activation so we just use the identity
        self._output_feedforward = FeedForward(hidden_dims[-1], 1, 1, lambda x: x)
        
        self._accuracy = BooleanAccuracy()
        
    def _get_hidden(self, output):
        layers = list(zip(self._linear_layers, self._activations,
                          self._dropout, self._gate_layers))
        layer, activation, dropout, _ = layers[0]
        output = dropout(activation(layer(output)))
        
        for layer, activation, dropout, gate in layers[1:]:
            gate_output = torch.sigmoid(gate(output))        
            new_output = dropout(activation(layer(output)))

            output = torch.add(torch.mul(gate_output, new_output),
                               torch.mul(1-gate_output, output))
            
        return output
    
    def forward(self,
                real_output,
                fake_output=None):

        real_hidden = self._get_hidden(real_output)

        real_value = self._output_feedforward(real_hidden)
        labels = torch.ones(real_hidden.size(0))
        if torch.cuda.is_available() and real_value.is_cuda:
            idx = real_value.get_device()
            labels = labels.cuda(idx)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(real_value.view(-1), labels)
                                                
        predictions = torch.sigmoid(real_value) > 0.5
        
        if fake_output is not None:
            fake_hidden = self._get_hidden(fake_output)
            fake_value = self._output_feedforward(fake_hidden)
            fake_labels = torch.zeros(fake_hidden.size(0))

            if torch.cuda.is_available() and fake_value.is_cuda:
                idx = fake_value.get_device()
                fake_labels = fake_labels.cuda(idx)
                        
            loss += torch.nn.functional.binary_cross_entropy_with_logits(fake_value.view(-1),
                                                                         fake_labels)
                                                            
            predictions = torch.cat([predictions, torch.sigmoid(fake_value) > 0.5])
            labels = torch.cat([labels, fake_labels])

        self._accuracy(predictions, labels.byte())
                    
        return {'loss': loss, 'predictions': predictions, 'labels': labels}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
