import typing
from typing import Optional

import torch

from allennlp.common import Params

from allennlp.models.model import Model

from allennlp.modules.attention import DotProductAttention
from allennlp.modules import MatrixAttention, FeedForward, Seq2SeqEncoder, SimilarityFunction, InputVariationalDropout, Seq2VecEncoder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention

from allennlp.nn.util import weighted_sum, masked_softmax, replace_masked_values

from . import IntraAttention, PoolingEncoder

class InterAttentionEncoder(Seq2VecEncoder):
    def forward(response, response_mask, source, source_mask):
        raise NotImplementedError

    '''
    @classmethod
    def from_params(cls, params: Params) -> 'InterAttentionEncoder':
        attention_type = params.pop('type')
        return type_lookup[attention_type].from_params(params=params)
    '''
    
@Seq2VecEncoder.register("combined_pooling_encoder")
class CombinedPoolingEncoder(InterAttentionEncoder):
    def __init__(self,
                 source_pooling_encoder: PoolingEncoder,
                 response_pooling_encoder: PoolingEncoder,
                 combine_feedforward: Optional[FeedForward] = None):

        super().__init__()
        
        self._source_pooling_encoder = source_pooling_encoder
        self._response_pooling_encoder = response_pooling_encoder
        self._combine_feedforward = combine_feedforward

    def forward(self, response, response_mask, source, source_mask):
        pooled_response = self._response_pooling_encoder(response, response_mask)
        pooled_source = self._source_pooling_encoder(source, source_mask)

        combined = torch.cat([pooled_response, pooled_source], dim=-1)
        if self._combine_feedforward is not None:
            combined = self._combine_feedforward(combined)
        return combined
        
@Seq2VecEncoder.register("memory_attention")
class MemoryAttention(InterAttentionEncoder):
    def __init__(self, attention: IntraAttention,
                 source_encoder: Seq2VecEncoder,
                 #input_dim: int,
                 memory_feedforward: Optional[FeedForward] = None,                 
                 n_hops: int = 3):
        
        super().__init__()
        self.n_hops = n_hops
        #self.query = torch.nn.Parameter(torch.Tensor(input_dim))
        #TODO: separate query for each hop?
        self._attention = attention
        self._source_encoder = source_encoder
        self._memory_feedforward = memory_feedforward
        
    def forward(self, response, response_mask, source, source_mask):
        #source is batch_size x n_sentences x n_dim
        #response is batch_size x n_sentences x n_dim
        #masks are batch_size x n_sentences

        batch_size, n_sentences, n_dim = response.shape
        
        #source is now batch_size x n_dim -> b x n_sentences x n-dim
        source = self._source_encoder(source, source_mask)                                    

        #initialize memory with average of response sentences        
        weighted_response = torch.sum(response * response_mask.float().unsqueeze(-1),
                                      dim=1) / torch.sum(response_mask.float(), 1, keepdim=True)

        #print(source.shape, response.shape, weighted_response.shape)
        for _ in range(self.n_hops):
            memory = torch.cat([response,
                                source.unsqueeze(1).expand_as(response),
                                weighted_response.unsqueeze(1).expand_as(response)],
                                dim=-1)
            
            response_attention = self._attention(memory, response_mask)
            if self._memory_feedforward is None:
                weighted_response = weighted_sum(response, response_attention)
            else:
                weighted_response = self._memory_feedforward(weighted_sum(memory, response_attention))
            
        return weighted_response


class ConditionalSeq2SeqEncoder(Seq2SeqEncoder):
    pass
    
@ConditionalSeq2SeqEncoder.register("esim_encoder")
class ESIMEncoder(ConditionalSeq2SeqEncoder):
    def __init__(self,    
                 similarity_function: SimilarityFunction,
                 response_projection_feedforward: FeedForward,
                 response_inference_encoder: Seq2SeqEncoder,
                 response_input_feedforward: Optional[FeedForward] = None,                 
                 source_input_feedforward: Optional[FeedForward] = None,
                 source_projection_feedforward: Optional[FeedForward] = None,                 
                 source_inference_encoder: Optional[Seq2SeqEncoder] = None,                 
                 dropout: float = 0.5) -> None:
        
        super().__init__()

        self._response_input_feedforward = response_input_feedforward
        self._response_projection_feedforward = response_projection_feedforward
        self._response_inference_encoder = response_inference_encoder

        self._source_input_feedforward = source_input_feedforward or response_input_feedforward
        self._source_projection_feedforward = source_projection_feedforward or response_projection_feedforward
        self._source_inference_encoder = source_inference_encoder or response_inference_encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        
        if dropout:
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.rnn_input_dropout = None

    def forward(self, encoded_response, response_mask,
                encoded_source, source_mask,
                #whether to only consider the response and alignments from the source to response
                response_only=False,
                #whether to concatenate the inputs to the outputs
                pass_input_through=False):

        response_output = []
        if pass_input_through:
            response_output = [encoded_response]
                    
        source_output = []
        if pass_input_through and not response_only:
            source_output = [encoded_source]
            
        if self._source_input_feedforward:
            encoded_source = self._source_input_feedforward(encoded_source)
        if self._response_input_feedforward:
            encoded_response = self._response_input_feedforward(encoded_response)
        
        # Shape: (batch_size, source_length, response_length)
        similarity_matrix = self._matrix_attention(encoded_source, encoded_response)

        # Shape: (batch_size, response_length, source_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), source_mask)
        # Shape: (batch_size, response_length, embedding_dim)
        attended_source = weighted_sum(encoded_source, h2p_attention)
            
        response_enhanced = torch.cat(
            [encoded_response, attended_source,
            encoded_response - attended_source,
            encoded_response * attended_source],
            dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_response = self._response_projection_feedforward(response_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_response = self.rnn_input_dropout(projected_enhanced_response)
        v_bi = self._response_inference_encoder(projected_enhanced_response, response_mask)

        response_output.append(v_bi)
        response_output = torch.cat(response_output, dim=-1)
        
        if not response_only:
            # Shape: (batch_size, source_length, response_length)
            p2h_attention = masked_softmax(similarity_matrix, response_mask)
            # Shape: (batch_size, source_length, embedding_dim)
            attended_response = weighted_sum(encoded_response, p2h_attention)
        
            # the "enhancement" layer
            source_enhanced = torch.cat(
                [encoded_source, attended_response,
                encoded_source - attended_response,
                encoded_source * attended_response],
                dim=-1
            )

            # The projection layer down to the model dimension.  Dropout is not applied before
            # projection.
            projected_enhanced_source = self._source_projection_feedforward(source_enhanced)
            
            if self.rnn_input_dropout:
                projected_enhanced_source = self.rnn_input_dropout(projected_enhanced_source)

            v_ai = self._source_inference_encoder(projected_enhanced_source, source_mask)

            source_output.append(v_ai)
            source_output = torch.cat(source_output, dim=-1)
        else:
            source_output = None
                          
        return response_output, source_output

type_lookup = dict(memory_attention=MemoryAttention,
                   combined_pooling_encoder=CombinedPoolingEncoder)    

