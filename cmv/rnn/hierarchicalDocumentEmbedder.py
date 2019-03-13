import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, InputVariationalDropout

from allennlp.nn.util import get_text_field_mask, weighted_sum

from cmv.rnn.cmvExtractor import extract
from cmv.rnn.attention.interAttention import ConditionalSeq2SeqEncoder

#TODO: modify seq2vec encoder to take in sentence features
#TODO: allow model to take encoded sentence, encoded aligned contextual sentence, and memory-weighted doc rep

@Model.register("hierarchical_document_embedder")
class HierarchicalDocumentEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 sentence_encoder: Seq2SeqEncoder,
                 dropout: float = 0.5):

        super().__init__(vocab=vocab)
        
        self._embedder = embedder
        self._encoder = encoder
        self._sentence_encoder = sentence_encoder

    def forward(self,
                post,
                features=None,
                idxs=None):

        embedded = self._embedder(post, num_wrapping_dims=1)
        mask = get_text_field_mask(post, num_wrapping_dims=1)

        batch_size, max_doc_len, max_sent_len = mask.shape

        #print(embedded.shape, mask.shape)                
        #reshape to be batch_size * n_sentences x n_words x dim?
        sentence_encoded = self._encoder(embedded.view(batch_size*max_doc_len,
                                                       max_sent_len, -1),
                                         mask.view(batch_size*max_doc_len, -1))
        #print(sentence_encoded.shape)
        
        sentence_mask = mask.sum(dim=-1) > 0

        #print(sentence_mask.shape)
        document = self._sentence_encoder(sentence_encoded.view(batch_size, max_doc_len, -1),
                                          sentence_mask)

        #print(document.shape)
        return document, sentence_mask

@Model.register("hierarchical_document_embedder")
class HierarchicalDocumentEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 sentence_encoder: Seq2SeqEncoder,
                 dropout: float = 0.5,
                 feature_feedforward: FeedForward = None):

        super().__init__(vocab=vocab)
        
        self._embedder = embedder
        self._encoder = encoder
        self._sentence_encoder = sentence_encoder
        self._feature_feedforward = feature_feedforward

        if dropout:
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.rnn_input_dropout = None
                
    def forward(self,
                post,
                features=None,
                idxs=None):

        if idxs is not None:
            post = extract(post, idxs)
            features = extract(post, idxs, features)
            
        embedded = self._embedder(post, num_wrapping_dims=1)
        mask = get_text_field_mask({i:j for i,j in response.items() if i!='mask'},
                                   num_wrapping_dims=1)

        # apply dropout for LSTM        
        if self.rnn_input_dropout:            
            embedded = self.rnn_input_dropout(embedded)

        #reshape to be batch_size * n_sentences x n_words x dim?
        batch_size, max_doc_len, max_sent_len = mask.shape

        # encode response at sentence level                                    
        sentence_encoded = self._encoder(embedded.view(batch_size*max_doc_len,
                                                       max_sent_len, -1),
                                         mask.view(batch_size*max_doc_len, -1))

        #before sentences, append features to each sentence
        sentence_features = [sentence_encoded_response]
        if response_features is not None:
            sentence_features.append(response_features)                
        if self._feature_feedforward is not None:
            sentence_encoded_response = self._feature_feedforward(torch.cat(sentence_features, dim=-1))        
                
        sentence_mask = mask.sum(dim=-1) > 0

        document = self._sentence_encoder(sentence_encoded.view(batch_size, max_doc_len, -1),
                                          sentence_mask)

        return document, sentence_mask
                
@Model.register("conditional_hierarchical_document_embedder")
class ConditionalHierarchicalDocumentEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_document_embedder: HierarchicalDocumentEmbedder,
                 interaction_encoder: ConditionalSeq2SeqEncoder,
                 response_document_embedder: Optional[HierarchicalDocumentEmbedder] = None,                 
                 dropout: float = 0.5):

        super().__init__(vocab=vocab)
                
        self._source_document_embedder = source_document_embedder
        self._response_document_embedder = response_document_embedder
        if response_document_embedder is None:
            self._response_document_embedder = source_document_embedder
        self._interaction_encoder = interaction_encoder

   def forward(self,
               source,
               response,
               source_features=None,
               response_features=None,
               idxs=None,
               compress_response=True):

       encoded_source,source_mask = self._source_document_embedder(source, source_features,
                                                        None if compress_response else idxs)
       encoded_response,response_mask = self._response_document_embedder(response, response_features,
                                                       idxs if compress_response else None)
       encoded_source_response_interaction, encoded_response_source_interaction = self._interaction_encoder(encoded_source,
                                                                                                            source_mask,
                                                                                                            encoded_response,
                                                                                                            response_mask)

       return encoded_source_response_interaction, encoded_response_source_interaction
