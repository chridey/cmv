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

from cmv.rnn.attention import InterAttentionEncoder, QueryAttentionEncoder, ConditionalSeq2SeqEncoder
from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvExtractor import extract

@Model.register("cmv_predictor")
class CMVPredictor(Model):
    def __init__(self, vocab: Vocabulary,
                 response_sentence_encoder: HierarchicalDocumentEmbedder,
                 response_encoder: Seq2VecEncoder,
                 output_feedforward: FeedForward,
                 interaction_encoder: Optional[ConditionalSeq2SeqEncoder] = None,                 
                 op_sentence_encoder: Optional[HierarchicalDocumentEmbedder] = None,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)

        self._response_sentence_encoder = response_sentence_encoder
        self._response_encoder = response_encoder        
        self._op_sentence_encoder = op_sentence_encoder or response_sentence_encoder

        self._interaction_encoder = interaction_encoder
             
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

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
                response: Dict[str, torch.LongTensor] = None,
                label: torch.IntTensor = None,
                original_post: Optional[Dict[str, torch.LongTensor]] = None,
                weakpoints: Optional[torch.LongTensor]=None,
                fake_data: bool=False,
                idxs: Optional[torch.LongTensor]=None,
                op_features: list=None,
                response_features: list=None,
                compress_response: bool=False,
                op_doc_features: list=None,
                response_doc_features: list=None,
                goodpoints=None
                ) -> Dict[str, torch.Tensor]:

        #print(original_post)
        #print(response)
        #print('label', label)        

        '''
        print('LABEL', label[0])
        for key in original_post:
            print('ORIGINAL POST')            
            for i in range(original_post[key][0].size(0)):
                o = [self.vocab.get_token_from_index(int(index), key) for index in original_post[key][0][i] if int(index)]
                if len(o):
                    print(o)
            print('RESPONSE') 
            for i in range(response[key][0].size(0)):
                o = [self.vocab.get_token_from_index(int(index), key) for index in response[key][0][i] if int(index)]
                if len(o):
                    print(o)
        '''

        output_dict = {}    
        encoded_response = None
        response_mask = None
        combined_input = []
        if response is not None or response_features is not None:
            sentence_encoded_response, response_mask = self._response_sentence_encoder(response,
                                                                            response_features,
                                                                            idxs if compress_response else None)

            if original_post is not None or op_features is not None:
                
                sentence_encoded_op, op_mask = self._op_sentence_encoder(original_post,
                                                                             op_features,
                                                                             None if compress_response else idxs)


                if self._interaction_encoder is not None:
                    output_dict.update({'pre_interaction_encoded_response':sentence_encoded_response})
                    sentence_encoded_response, sentence_encoded_op  = self._interaction_encoder(sentence_encoded_response,
                                                                                               response_mask,
                                                                                               sentence_encoded_op,
                                                                                               op_mask)
                
                pooled_response = self._response_encoder(sentence_encoded_response, response_mask,
                                                         sentence_encoded_op, op_mask)
            else:
                pooled_response = self._response_encoder(sentence_encoded_response, response_mask)

            if self.dropout:
                pooled_response = self.dropout(pooled_response)

            combined_input = [pooled_response]

        if op_doc_features is not None:
            combined_input.append(op_doc_features)
        if response_doc_features is not None:
            combined_input.append(response_doc_features)

        combined_input = torch.cat(combined_input, dim=-1)
            
        label_logits = self._output_feedforward(combined_input).squeeze(-1)
        label_probs = torch.sigmoid(label_logits)
        #print(label_probs)
        predictions = label_probs > 0.5
        #print('predictions', predictions)
        #print('1-predictions', 1-predictions)
        #print(label)
        
        output_dict.update({"label_logits": label_logits, "label_probs": label_probs,
                        "representation": combined_input,
                        "encoded_response": sentence_encoded_response,
                        "response_mask": response_mask})

        if label is not None:
            true_weight = 1 if ((label==1).sum().float() == 0) else ((label==0).sum().float() / (label==1).sum().float())            
            #true_weight = (label==0).sum().float() / (label==1).sum().float()
            #print(true_weight)

            weight = label.eq(0).float() + label.eq(1).float() * true_weight
            loss = self._loss(label_logits, label.float(), weight=weight)

            if fake_data:
                self._fake_accuracy(predictions, label.byte())
                self._fake_fscore(torch.stack([1-predictions, predictions], dim=1), label)
            else:
                self._accuracy(predictions, label.byte())
                #self._cat_accuracy(torch.stack([1-predictions, predictions], dim=1), label.byte())
                self._fscore(torch.stack([1-predictions, predictions], dim=1), label)

            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ret = {}
        if self._accuracy._total_count:
            precision, recall, f1_measure = self._fscore.get_metric(reset)
            ret = {'accuracy': self._accuracy.get_metric(reset),
                #'cat_accuracy': self._cat_accuracy.get_metric(reset=True),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}

        if self._fake_accuracy._total_count == 0:
            return ret
        fake_accuracy = self._fake_accuracy.get_metric(reset)
        if fake_accuracy == 0:
            return ret

        fake_precision, fake_recall, fake_f1_measure = self._fake_fscore.get_metric(reset)
        ret.update({'fake_acc': fake_accuracy,
                    'fake_prec': fake_precision,
                    'fake_rec': fake_recall,
                    'fake_f1': fake_f1_measure})
        return ret
            

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
