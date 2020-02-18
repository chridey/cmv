import typing
from typing import Optional

import torch

from allennlp.common import Params

from allennlp.models.model import Model

from allennlp.modules.attention import DotProductAttention
from allennlp.modules import MatrixAttention, FeedForward, Seq2SeqEncoder, SimilarityFunction, InputVariationalDropout, Seq2VecEncoder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention

from allennlp.nn.util import weighted_sum, masked_softmax, replace_masked_values

from . import IntraAttention

class InterAttentionEncoder(Seq2VecEncoder):
    def forward(source, response, source_mask, response_mask, source_attention=None):
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'InterAttentionEncoder':
        attention_type = params.pop('type')
        return type_lookup[attention_type].from_params(params=params)
    
@Seq2VecEncoder.register("memory_attention")
class MemoryAttention(InterAttentionEncoder):
    def __init__(self, attention: IntraAttention,
                 memory_feedforward: FeedForward,
                 input_dim: int, n_hops: int = 3):
        
        super().__init__()
        self.n_hops = n_hops
        self.query = torch.nn.Parameter(torch.Tensor(input_dim))
        torch.nn.init.uniform(self.query, -1e-2, 1e-2)
        
        #TODO: separate query for each hop?
        self._attention = attention
        self._memory_feedforward = memory_feedforward
        
    def forward(self, source, response, source_mask, response_mask, source_attention=None):
        #source is batch_size x n_sentences x n_dim
        #response is batch_size x n_sentences x n_dim
        #masks are batch_size x n_sentences

        batch_size, n_sentences, n_dim = response.shape
        
        #source is now batch_size x n_dim -> b x n_sentences x n-dim
        if source_attention is not None:
            attention = source_attention(source, source_mask)
            #print(source.shape, attention.shape)
            source = weighted_sum(source, attention)
            #print(source.shape)
        else:
            source_mask = source_mask.float()
            source = torch.sum(source * source_mask.unsqueeze(-1), dim=1) / torch.sum(
                                source_mask, 1, keepdim=True)                                     

        #initialize memory with average of response sentences        
        weighted_response = torch.sum(response * response_mask.float().unsqueeze(-1), dim=1) / torch.sum(response_mask.float(), 1, keepdim=True)                                     

        #print(source.shape, response.shape, weighted_response.shape)
        for _ in range(self.n_hops):
            memory = torch.cat([response,
                                source.unsqueeze(1).expand_as(response),
                                weighted_response.unsqueeze(1).expand_as(response)],
                                dim=-1)

            response_attention = self._attention(memory, response_mask)
            #weighted_response = self._memory_feedforward(weighted_sum(memory, response_attention))
            weighted_response = weighted_sum(response, response_attention)
            
        return weighted_response

    @classmethod
    def from_params(cls, params: Params) -> 'MemoryAttention':
        attention = IntraAttention.from_params(params.pop('attention'))
        memory_feedforward = FeedForward.from_params(params.pop('memory_feedforward'))
        n_hops = params.pop('n_hops', 3)
        
        return cls(attention, memory_feedforward, n_hops)

@Model.register("conditional_seq2seq_encoder")
class ConditionalSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 input_feedforward: Optional[FeedForward] = None,                 
                 source_input_feedforward: Optional[FeedForward] = None,
                 source_projection_feedforward: Optional[FeedForward] = None,                 
                 source_inference_encoder: Optional[Seq2SeqEncoder] = None,                 
                 dropout: float = 0.5,
                 #whether to only consider the response and alignments from the source to response
                 response_only=False) -> None:
        
        super().__init__()

        self._response_input_feedforward = response_input_feedforward
        self._response_projection_feedforward = response_projection_feedforward
        self._response_inference_encoder = response_inference_encoder

        self._source_input_feedforward = source_input_feedforward or response_input_feedforward
        self._source_projection_feedforward = source_projection_feedforward or response_projection_feedforward
        self._source_inference_encoder = source_inference_encoder or response_inference_encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None
        
    def forward(self,
                encoded_source,
                encoded_response):
        pass         
    
@Seq2VecEncoder.register('esim_attention')
class ESIMAttention(InterAttentionEncoder):
    def __init__(self,
                 similarity_function: SimilarityFunction,
                 response_projection_feedforward: FeedForward,
                 response_inference_encoder: Seq2SeqEncoder,
                 response_input_feedforward: Optional[FeedForward] = None,                 
                 source_input_feedforward: Optional[FeedForward] = None,
                 source_projection_feedforward: Optional[FeedForward] = None,                 
                 source_inference_encoder: Optional[Seq2SeqEncoder] = None,                 
                 dropout: float = 0.5,
                 #whether to only consider the response and alignments from the source to response
                 response_only=False) -> None:
        
        super().__init__()

        self._response_input_feedforward = response_input_feedforward
        self._response_projection_feedforward = response_projection_feedforward
        self._response_inference_encoder = response_inference_encoder

        self._source_input_feedforward = source_input_feedforward or response_input_feedforward
        self._source_projection_feedforward = source_projection_feedforward or response_projection_feedforward
        self._source_inference_encoder = source_inference_encoder or response_inference_encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._response_only = response_only

    #def forward(self, encoded_source, encoded_response,
    #            source_mask, response_mask, source_attention=None):
    def forward(self, encoded_response, response_mask,
                encoded_source, source_mask):

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

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_b_max, _ = replace_masked_values(
            v_bi, response_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        
        v_b_avg = torch.sum(v_bi * response_mask.float().unsqueeze(-1), dim=1) / torch.sum(
            response_mask.float(), 1, keepdim=True
        )

        to_cat = [v_b_avg, v_b_max]        
        if not self._response_only:
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
            v_a_max, _ = replace_masked_values(
                v_ai, source_mask.unsqueeze(-1), -1e7
            ).max(dim=1)
            v_a_avg = torch.sum(v_ai * source_mask.float().unsqueeze(-1), dim=1) / torch.sum(
                source_mask.float(), 1, keepdim=True
            )
            to_cat = [v_a_avg, v_a_max] + to_cat
      
        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat(to_cat, dim=1)    

        #TODO
        #instead of max and average pooling, take a memory network
        #for OP, start with avg or max of OP and RR sentences
        #for RR, start with mem of OP and avg or max of RR sentences
        
        return v_all

    @classmethod
    def from_params(cls, params: Params) -> 'ESIMAttention':

        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        response_projection_feedforward = FeedForward.from_params(params.pop("response_projection_feedforward"))
        response_inference_encoder = Seq2SeqEncoder.from_params(params.pop("response_inference_encoder"))
               
        source_projection_feedforward_params = params.pop("source_projection_feedforward", None)
        source_projection_feedforward = None
        if source_projection_feedforward_params is not None:
            source_projection_feedforward = FeedForward.from_params(source_projection_feedforward_params)

        response_input_feedforward_params = params.pop("response_input_feedforward", None)
        response_input_feedforward = None
        if response_input_feedforward_params is not None:
            response_input_feedforward = FeedForward.from_params(response_input_feedforward_params)

        source_input_feedforward_params = params.pop("source_input_feedforward", None)
        source_input_feedforward = None
        if source_input_feedforward_params is not None:
            source_input_feedforward = FeedForward.from_params(source_input_feedforward_params)
                        
        source_inference_encoder_params = params.pop("source_inference_encoder", None)
        if source_inference_encoder_params is not None:
            source_inference_encoder = Seq2SeqEncoder.from_params(source_inference_encoder_params)
        else:
            source_inference_encoder = None
                                                
        dropout = params.pop("dropout", 0.5)
        
        return cls(similarity_function, response_projection_feedforward, response_inference_encoder,
                   response_input_feedforward, source_input_feedforward,
                   source_projection_feedforward, source_inference_encoder, dropout)
            
type_lookup = dict(memory_attention=MemoryAttention,
                   esim_attention=ESIMAttention)    
