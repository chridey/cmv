from typing import Optional
from overrides import overrides

import torch

from allennlp.common import Params

from allennlp.models.model import Model

from allennlp.modules import FeedForward
from allennlp.modules.attention import DotProductAttention
from allennlp.modules import Seq2VecEncoder

from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
class IntraAttention(DotProductAttention):
    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None) -> torch.Tensor:

        #print(matrix.shape, matrix_mask.shape)        
        similarities = self._forward_internal(self.query_vector(matrix, matrix_mask),
                                              self.matrix(matrix, matrix_mask))

        if self._normalize:
            similarities = masked_softmax(similarities, matrix_mask)

        return similarities

    @classmethod
    def from_params(cls, params: Params) -> 'IntraAttention':
        attention_type = params.pop('type')
        return type_lookup[attention_type].from_params(params=params)
        
@Model.register("query_attention")        
class QueryAttention(IntraAttention):
    def __init__(self, hidden_feedforward: FeedForward,
                 input_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.Tensor(input_dim))
        self.hidden_feedforward = hidden_feedforward
        
    def query_vector(self, matrix, matrix_mask):
        batch_size = matrix.size(0)
        return self.query.view(1, -1).expand(batch_size, -1)

    def matrix(self, matrix, matrix_mask):
        return self.hidden_feedforward(matrix)
    
    @classmethod
    def from_params(cls, params: Params) -> 'QueryAttention':
        input_dim = params.pop('input_dim')
        hidden_feedforward = FeedForward.from_params(params['hidden_feedforward'])
        return cls(hidden_feedforward, input_dim)    

@Seq2VecEncoder.register("pooling_encoder")    
class PoolingEncoder(Seq2VecEncoder):
    def __init__(self,
                 hidden_feedforward: Optional[FeedForward] = None,
                 projection_feedforward: Optional[FeedForward] = None,                 
                 max_pool = True,
                 avg_pool = True) -> None:

        super().__init__()
        self._hidden_feedforward = hidden_feedforward
        self._projection_feedforward = projection_feedforward        
        self._max_pool = max_pool
        self._avg_pool = avg_pool
        
    def forward(self,
                embedded_input,
                input_mask,
                other_input=None,
                other_mask=None):

        #assumes input is batch_size * num_words * embedding_dim
        
        if self._hidden_feedforward is not None:
            embedded_input = self._hidden_feedforward(embedded_input)

        to_cat = []
        if self._max_pool:
            input_max, _ = replace_masked_values(
                embedded_input, input_mask.unsqueeze(-1), -1e7
            ).max(dim=1)
            to_cat.append(input_max)
        if self._avg_pool:
            input_avg = torch.sum(embedded_input * input_mask.float().unsqueeze(-1),
                                dim=1) / torch.sum(
                    input_mask.float(), 1, keepdim=True
                )
            to_cat.append(input_avg)

        output = torch.cat(to_cat, dim=1)

        if self._projection_feedforward is not None:
            output = self._projection_feedforward(output)
        
        return output
    
@Seq2VecEncoder.register("query_attention_encoder")
class QueryAttentionEncoder(Seq2VecEncoder):
    def __init__(self,
                 query_attention: QueryAttention):

        super().__init__()
        self._query_attention = query_attention

    def forward(self,
                embedded_input,
                input_mask,
                other_input=None,
                other_mask=None):
                
        #get weighted average of words in sentence
        attention = self._query_attention(embedded_input, input_mask)
        encoded_input = weighted_sum(embedded_input, attention)

        return encoded_input
    
type_lookup = dict(query_attention=QueryAttention)
