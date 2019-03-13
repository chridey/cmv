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

@Model.register("cmv_predictor")
class CMVPredictor(Model):
    def __init__(self, vocab: Vocabulary,
                 response_embedder: TextFieldEmbedder,
                 response_word_attention: Seq2VecEncoder,
                 response_encoder: Seq2SeqEncoder,
                 response_sentence_attention: Seq2VecEncoder,
                 output_feedforward: FeedForward,
                 op_embedder: Optional[TextFieldEmbedder] = None,
                 op_word_attention: Optional[Seq2VecEncoder] = None,                 
                 op_encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 feature_feedforward: FeedForward = None) -> None:

        super().__init__(vocab, regularizer)

        self._response_embedder = response_embedder
        self._response_word_attention = response_word_attention
        self._response_encoder = response_encoder
        self._response_sentence_attention = response_sentence_attention
        
        self._op_embedder = op_embedder or response_embedder        
        self._op_word_attention = op_word_attention or response_word_attention
        self._op_encoder = op_encoder or response_encoder
             
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._feature_feedforward = feature_feedforward

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
                response_features: list=None,
                compress_response: bool=False,
                op_doc_features: list=None,
                response_doc_features: list=None,
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

        if False:
            encoded_response = None
            response_mask = None
            combined_input = []
        else:
            if idxs is not None and compress_response:
                response = extract(response, idxs)

            #print([(key, response[key].shape if response[key] is not None else None) for key in response])
            embedded_response = self._response_embedder(response, num_wrapping_dims=1)
            #print(embedded_op.shape, embedded_response.shape)
            response_mask = get_text_field_mask({i:j for i,j in response.items() if i!='mask'},
                                                num_wrapping_dims=1).float()

            if self.rnn_input_dropout:            
                embedded_response = self.rnn_input_dropout(embedded_response)

            # encode response at sentence level
            #print(response_mask.shape, embedded_response.shape)
            batch_size, max_response_sentences, max_response_words, response_dim = embedded_response.shape
            embedded_response = embedded_response.view(batch_size*max_response_sentences,
                                                       max_response_words, -1)
            response_mask = response_mask.view(batch_size*max_response_sentences,
                                               max_response_words)              
            sentence_encoded_response = self._response_word_attention(embedded_response,
                                                                      response_mask).view(batch_size,
                                                                                    max_response_sentences,
                                                                                    -1)

            #print(embedded_op.shape, op_mask.shape, embedded_response.shape, response_mask.shape)
            # apply dropout for LSTM

            #before sentences, append features to each sentence
            sentence_features = [sentence_encoded_response]
            if response_features is not None:
                sentence_features.append(response_features)                
            if self._feature_feedforward is not None:
                #TODO: need to compress response features also? why doesnt this cause an error - padding
                
                #print(sentence_encoded_response.shape, response_features.shape)
                sentence_encoded_response = self._feature_feedforward(torch.cat(sentence_features, dim=-1))

            response_mask = response_mask.view(batch_size, max_response_sentences, -1).sum(dim=-1) > 0
            encoded_response = self._response_encoder(sentence_encoded_response, response_mask)

            if original_post is not None:
                if idxs is not None and not compress_response:
                    original_post = extract(original_post, idxs)
                    
                embedded_op = self._op_embedder(original_post, num_wrapping_dims=1)        
                op_mask = get_text_field_mask({i:j for i,j in original_post.items() if i!='mask'},
                                            num_wrapping_dims=1).float()

                # apply dropout for LSTM        
                if self.rnn_input_dropout:
                    embedded_op = self.rnn_input_dropout(embedded_op)

                batch_size, max_op_sentences, max_op_words, op_dim = embedded_op.shape
                embedded_op = embedded_op.view(batch_size*max_op_sentences,
                                                        max_op_words, -1)
                op_mask = op_mask.view(batch_size*max_op_sentences,
                                                max_op_words)              
                sentence_encoded_op = self._op_word_attention(embedded_op,
                                                              op_mask).view(batch_size,
                                                                            max_op_sentences,
                                                                            -1)

                sentence_features = [sentence_encoded_op]
                if op_features is not None:
                    sentence_features.append(op_features)                
                if self._feature_feedforward is not None:
                    sentence_encoded_op = self._feature_feedforward(torch.cat(sentence_features, dim=-1))
                    
                op_mask = op_mask.view(batch_size, max_op_sentences, -1).sum(dim=-1) > 0
                encoded_op = self._op_encoder(sentence_encoded_op, op_mask)

                #if idxs is not None and not compress_response:
                #    encoded_op = extract({'tokens': encoded_op}, idxs)['tokens']
                #    extracted_sentences = extract(original_post, idxs)
                #    op_mask = (get_text_field_mask(extracted_sentences,
                #                                  num_wrapping_dims=1).sum(dim=-1) > 0)

            else:
                op_mask = None
                encoded_op = None            

            #if idxs is not None and compress_response:
            #    encoded_response = extract({'tokens': encoded_response}, idxs)['tokens']
            #    extracted_sentences = extract(response, idxs)
            #    response_mask = (get_text_field_mask(extracted_sentences,
            #                                  num_wrapping_dims=1).sum(dim=-1) > 0)

            #print(encoded_response.shape, response_mask.shape, encoded_op.shape, op_mask.shape)
            combined_input = self._response_sentence_attention(encoded_response, response_mask,
                                                               encoded_op, op_mask)                    

            #now batch_size x n_dim
            #encoded_op = self._op_sentence_attention(encoded_op, op_mask)

            if self.dropout:
                combined_input = self.dropout(combined_input)

            combined_input = [combined_input]

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
        
        output_dict = {"label_logits": label_logits, "label_probs": label_probs,
                       "representation": combined_input,
                       "encoded_response": encoded_response,
                       "response_mask": response_mask}

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
            
    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary) -> 'CMVPredictor':

        response_embedder_params = params.pop("response_embedder")
        response_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=response_embedder_params)

        response_word_attention_params = params.pop("response_word_attention")
        response_word_attention = Seq2VecEncoder.from_params(params=response_word_attention_params)

        response_encoder_params = params.pop("response_encoder")
        response_encoder = Seq2SeqEncoder.from_params(params=response_encoder_params)

        response_sentence_attention_params = params.pop("response_sentence_attention")
        response_sentence_attention = Seq2VecEncoder.from_params(params=response_sentence_attention_params)

        op_embedder_params = params.pop("op_embedder", None)
        op_embedder = None
        if op_embedder_params is not None:
            op_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=op_embedder_params)

        op_word_attention_params = params.pop("op_word_attention", None)
        op_word_attention = None
        if op_word_attention_params is not None:
            op_word_attention = Seq2VecEncoder.from_params(params=op_word_attention_params)

        op_encoder_params = params.pop("op_encoder", None)
        op_encoder = None
        if op_encoder_params is not None:
            op_encoder = Seq2SeqEncoder.from_params(params=op_encoder_params)

        output_feedforward = FeedForward.from_params(params=params.pop('output_feedforward'))

        feature_feedforward = None
        feature_feedforward_params=params.pop('feature_feedforward', None)
        if feature_feedforward_params is not None:
            feature_feedforward = FeedForward.from_params(params=feature_feedforward_params)
            
        dropout = params.pop("dropout", 0)

        initializer = InitializerApplicator.from_params(params=params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params=params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab,
                   response_embedder,
                   response_word_attention,
                   response_encoder,
                   response_sentence_attention,
                   output_feedforward,
                   op_embedder,
                   op_word_attention,
                   op_encoder,
                   dropout,
                   initializer,
                   regularizer,
                   feature_feedforward)

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
