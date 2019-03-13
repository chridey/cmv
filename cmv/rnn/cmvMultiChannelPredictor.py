from typing import Dict, Optional, List, Any

import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model

from allennlp.common.checks import check_dimensions_match
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward, InputVariationalDropout, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, BooleanAccuracy

from cmv.rnn.attention import InterAttentionEncoder, QueryAttentionEncoder
from cmv.rnn.cmvExtractor import extract

from cmv.rnn.cmvPredictor import CMVPredictor

@Model.register("cmv_multi_channel_predictor")
class CMVMultiChannelPredictor(CMVPredictor):
    def __init__(self,
                 vocab: Vocabulary,
                 response_only_predictor: CMVPredictor,
                 op_response_predictor: CMVPredictor,
                 output_feedforward: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        Model.__init__(self, vocab, regularizer)

        self._response_only_predictor = response_only_predictor
        self._op_response_predictor = op_response_predictor
             
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._output_feedforward = output_feedforward

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        #check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
        #                       "text field embedding dim", "encoder input dim")
        #check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
        #                       "encoder output dim", "projection feedforward input")
        #check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
        #                       "proj feedforward output dim", "inference lstm input dim")

        self._accuracy = BooleanAccuracy()
        self._fscore = F1Measure(positive_label=1)
        
        self._fake_accuracy = BooleanAccuracy()
        self._fake_fscore = F1Measure(positive_label=1)
        
        self._loss = torch.nn.functional.binary_cross_entropy_with_logits

        initializer(self)

    def forward(self,                
                response: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                original_post: Optional[Dict[str, torch.LongTensor]] = None,
                weakpoints: Optional[torch.IntTensor] = None,
                fake_data: bool=False,
                idxs: Optional[torch.LongTensor]=None,
                op_features: list=None,
                response_features: list=None
                ) -> Dict[str, torch.Tensor]:


        response_only_output = self._response_only_predictor(response, label, weakpoints=weakpoints, fake_data=fake_data, idxs=idxs, response_features=response_features)
        op_response_output = self._op_response_predictor(response, label, original_post, weakpoints, fake_data, idxs, op_features, response_features)
        combined_input = torch.cat([response_only_output['representation'], op_response_output['representation']], dim=-1)

        #the sentence representations are of shape BxSxD (assuming same dimension)
        batch_size, max_sentences, _ = response_only_output['encoded_response'].shape
        #print(response_only_output['response_mask'].shape, response_only_output['encoded_response'].shape)
        orthogonality_loss = response_only_output['response_mask'].float() * (response_only_output['encoded_response'] * op_response_output['encoded_response']).sum(dim=-1).abs().squeeze(-1)
        #print(orthogonality_loss.shape)
        orthogonality_loss_avg = torch.sum(orthogonality_loss, dim=1) / torch.sum(response_only_output['response_mask'].float())
        #print(orthogonality_loss_avg.shape)
        
        label_logits = self._output_feedforward(combined_input).squeeze(-1)
        label_probs = torch.sigmoid(label_logits)

        predictions = label_probs > 0.5
        
        output_dict = {"label_logits": label_logits, "label_probs": label_probs,
                       "representation": combined_input}

        true_weight = 1 if ((label==1).sum().float() == 0) else ((label==0).sum().float() / (label==1).sum().float())            
        
        weight = label.eq(0).float() + label.eq(1).float() * true_weight
        loss = self._loss(label_logits, label.float(), weight=weight)

        #print(loss, orthogonality_loss_avg.mean())
        
        if fake_data:
            self._fake_accuracy(predictions, label.byte())
            self._fake_fscore(torch.stack([1-predictions, predictions], dim=1), label)
        else:
            self._accuracy(predictions, label.byte())
            self._fscore(torch.stack([1-predictions, predictions], dim=1), label)
        
        output_dict["loss"] = loss + orthogonality_loss_avg.mean()

        return output_dict

    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary) -> 'CMVMultiChannelPredictor':

        response_only_predictor = Model.from_params(params=params.pop('response_only_predictor'), vocab=vocab)

        op_response_predictor = Model.from_params(params=params.pop('op_response_predictor'), vocab=vocab)
        
        output_feedforward = FeedForward.from_params(params=params.pop('output_feedforward'))
            
        dropout = params.pop("dropout", 0)

        initializer = InitializerApplicator.from_params(params=params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params=params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab,
                   response_only_predictor,
                   op_response_predictor,
                   output_feedforward,
                   dropout,
                   initializer,
                   regularizer)
        
            
'''
TODO
projection layer
max/min pool
ESIM type reps before memory network
frequency weighted embeddings similarity, then max/min/avg pool
structural features like paragraph breaks
SIF weighted embeddings

what is the extractor actually extracting?

why did the model start improving when i trained the predictor and discriminator jointly?

compare to 5 randomly extracted sentences
compare to quotes
compare to arg mining data

try different number of extracted sentences and learning when to stop

is the model just learning based on the intermediate discussion?
    try removing this during transformer training
fix extractor when training generator
instead of max pool do just avg or weighted avg so that model is forced to consider entire OP
rather than just features from response
transformers without positional encodings
separate predictors for compressed and full (except for classification layer)
   fix full predictor prior to training GAN
   also train extractor/predictor for compressed while fixing full predictor 
'''                       
