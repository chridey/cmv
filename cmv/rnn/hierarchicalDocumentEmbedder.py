from typing import Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, InputVariationalDropout, FeedForward

from allennlp.nn.util import get_text_field_mask, weighted_sum

from cmv.rnn.cmvExtractor import extract
from cmv.rnn.attention.interAttention import ConditionalSeq2SeqEncoder

#TODO: modify seq2vec encoder to take in sentence features
#TODO: allow model to take encoded sentence, encoded aligned contextual sentence, and memory-weighted doc rep

@Model.register("hierarchical_document_embedder")
class HierarchicalDocumentEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_encoder: Seq2SeqEncoder,
                 embedder: Optional[TextFieldEmbedder] = None,
                 encoder: Optional[Seq2VecEncoder] = None,                 
                 dropout: float = 0.5,
                 feature_feedforward: FeedForward = None,
                 compress_before=False):

        super().__init__(vocab=vocab)
        
        self._embedder = embedder
        self._encoder = encoder
        self._sentence_encoder = sentence_encoder
        self._feature_feedforward = feature_feedforward

        #if dropout:
        #    self.rnn_input_dropout = InputVariationalDropout(dropout)
        #else:
        self.rnn_input_dropout = None
        self._compress_before = compress_before
        
    def forward(self,
                post=None,
                features=None,
                idxs=None):

        if idxs is not None and self._compress_before:
            extracted_post = {}
            if post is not None:
                for key in post:
                    extracted_post[key] = extract(post[key], idxs)
                post = extracted_post
            features = extract(features, idxs)

        sentence_features = []
        if post is not None:
            embedded = self._embedder(post, num_wrapping_dims=1)
            mask = get_text_field_mask({i:j for i,j in post.items() if i!='mask'},
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

            sentence_encoded = sentence_encoded.view(batch_size, max_doc_len, -1)
            #before sentences, append features to each sentence
            sentence_features.append(sentence_encoded)

            sentence_mask = mask.sum(dim=-1) > 0

        else:
            #need to compute sentence mask from features
            sentence_mask = []
            for feature in features:
                i = 0
                for idx,sentence_feature in enumerate(feature):
                    if sentence_feature.sum() > 0:
                        i = idx
                sentence_mask.append([1]*i + [0]*(features.shape[1]-i))
            sentence_mask = torch.FloatTensor(sentence_mask)
            if torch.cuda.is_available() and features.is_cuda:
                idx = features.get_device()
                sentence_mask = sentence_mask.cuda(idx)
            
        if features is not None:
            sentence_features.append(features)
            
        if self._feature_feedforward is not None:
            sentence_encoded = self._feature_feedforward(torch.cat(sentence_features, dim=-1))        
        else:
            sentence_encoded = torch.cat(sentence_features, dim=-1)
        
        document = self._sentence_encoder(sentence_encoded,
                                          sentence_mask)

        if idxs is not None and not self._compress_before:
            document = extract(document, idxs)
            sentence_mask = extract(sentence_mask.unsqueeze(-1), idxs).squeeze(-1)
        
        return document, sentence_mask

'''                    
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
        encoded_source_response_interaction, encoded_response_source_interaction = self._interaction_encoder(encoded_response,
response_mask,
encoded_source,
source_mask)
    
        return encoded_source_response_interaction, encoded_response_source_interaction
'''
