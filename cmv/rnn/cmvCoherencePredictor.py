import collections

from typing import Dict, Optional, List, Any

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model

from allennlp.common.checks import check_dimensions_match
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward, InputVariationalDropout
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, BooleanAccuracy

from cmv.rnn.attention.intraAttention import IntraAttention
from cmv.rnn.attention.interAttention import InterAttentionEncoder

INI = 1e-2

@Model.register("cmv_coherence_predictor")
class CMVCoherencePredictor(Model):
    def __init__(self, vocab: Vocabulary,
                 response_encoder: Seq2SeqEncoder,
                 sentence_attention: IntraAttention,
                 coherence_output_feedforward: FeedForward,
                 global_output_feedforward: Optional[FeedForward] = None,
                 op_encoder: Optional[Seq2SeqEncoder] = None,
                 response_sentence_attention: Optional[InterAttentionEncoder] = None,                 
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 weights_file: str = None) -> None:

        super().__init__(vocab, regularizer)

        self._response_encoder = response_encoder
        self._sentence_attention = sentence_attention
        self._op_encoder = op_encoder or response_encoder
        self._response_sentence_attention = response_sentence_attention
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._coherence_output_feedforward = coherence_output_feedforward
        self._global_output_feedforward = global_output_feedforward        
        self._weight = torch.nn.Parameter(torch.Tensor(2))
        torch.nn.init.uniform(self._weight, -INI, INI)
            
        self._num_labels = 2

        self._accuracy = BooleanAccuracy()
        self._fscore = F1Measure(positive_label=1)
        
        self._loss = torch.nn.functional.binary_cross_entropy_with_logits

        initializer(self)

        if weights_file is not None:
            weights = torch.load(weights_file)
            state_dict = collections.defaultdict(dict)
            for key in weights:
                if key == '_weight':
                    continue
                if getattr(self, key.split('.')[0], None) is not None:
                    state_dict[key.split('.')[0]][key.split('.', 1)[1]] = weights[key]
            for module_name in state_dict:
                getattr(self, module_name).load_state_dict(state_dict[module_name])
                

    def forward(self,
                response,
                paragraphs,
                global_features,
                label: torch.IntTensor = None,
                original_post = None,
                op_paragraphs=None,                
                op_features: list=None,
                response_features: list=None,
                ) -> Dict[str, torch.Tensor]:


        #print(response['tokens'].shape)
        response_mask = get_text_field_mask(response,
                                            num_wrapping_dims=1).float()
        #print(response_mask.shape)
        batch_size, max_response_sentences, _ = response_mask.shape
        response_mask = response_mask.view(batch_size, max_response_sentences, -1).sum(dim=-1) > 0
        #print(response_mask.shape)        
        #print(response_mask)
        
        embedded_response = paragraphs
        # apply dropout for LSTM
        #if self.rnn_input_dropout:            
        #    embedded_response = self.rnn_input_dropout(embedded_response)        

        #print(embedded_response.shape)
        #print(embedded_response[0])
        encoded_response = self._response_encoder(embedded_response, response_mask)
        #print(encoded_response[0])
        #print(encoded_response.shape)

        if original_post is not None:
            op_mask = get_text_field_mask(original_post,
                                        num_wrapping_dims=1).float()
            _, max_op_sentences, _ = op_mask.shape
            op_mask = op_mask.view(batch_size, max_op_sentences, -1).sum(dim=-1) > 0

            embedded_op = op_paragraphs
            encoded_op = self._op_encoder(embedded_op, op_mask)

            combined_input = self._response_sentence_attention(encoded_op, encoded_response,
                                                            op_mask, response_mask,
                                                            self._sentence_attention)
                    
        else:
            attn = self._sentence_attention(encoded_response, response_mask)
            #print(attn[0])
            combined_input = weighted_sum(encoded_response, attn)
        #print(combined_input.shape)
        
        if self.dropout:
            combined_input = self.dropout(combined_input)
        #print(combined_input[0])
                                    
        coherence_logit = self._coherence_output_feedforward(combined_input)
        #print(coherence_logit.shape)
        #print(coherence_logit)
        
        #print(global_features.shape)
        if self._global_output_feedforward is not None:
            global_logit = self._global_output_feedforward(global_features)
        else:
            global_logit = global_features
        #print(global_logit.shape)
        #print(self._weight.shape)
        label_logits = coherence_logit.view(-1) #UNDO
        #label_logits = torch.matmul(torch.cat([coherence_logit, global_logit], dim=-1), torch.softmax(self._weight, dim=0))#
        if not self.training:
            print(torch.softmax(self._weight, dim=0))
        #print(label_logits.shape)
        label_probs = torch.sigmoid(label_logits)
        predictions = label_probs > 0.5
        
        output_dict = {"label_logits": label_logits, "label_probs": label_probs}
    
        true_weight = (label==0).sum().float() / (label==1).sum().float()
        #print(true_weight)
        
        weight = label.eq(0).float() + label.eq(1).float() * true_weight
        loss = self._loss(label_logits, label.float(), weight=weight)

        self._accuracy(predictions, label.byte())
        self._fscore(torch.stack([1-predictions, predictions], dim=1), label)
        
        output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self._fscore.get_metric(reset)
        ret = {'accuracy': self._accuracy.get_metric(reset),
               'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}

        return ret
            
