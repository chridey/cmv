from typing import Dict

import torch

from allennlp.models import Model
from allennlp.training.metrics import BooleanAccuracy

from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvExtractor import CMVExtractor, extract
from cmv.rnn.cmvDiscriminator import CMVDiscriminator
from cmv.rnn.cmvActorCritic import CMVActorCritic


@Model.register("cmv_discriminator_trainer")
class CMVDiscriminatorTrainer(Model):
    def __init__(self,
                 document_embedder: HierarchicalDocumentEmbedder,
                 cmv_predictor: CMVPredictor,
                 cmv_extractor: CMVExtractor,
                 cmv_discriminator: CMVDiscriminator,
                 compress_response: bool=False):

        super().__init__(vocab=None)

        self._document_embedder = document_embedder
        
        self._cmv_predictor = cmv_predictor
        self._cmv_extractor = cmv_extractor
        self._cmv_discriminator = cmv_discriminator
        self._compress_response = compress_response
        
    def forward(self,
                response: Dict[str, torch.LongTensor],
                label: torch.IntTensor,
                original_post: Dict[str, torch.LongTensor],
                weakpoints=None,
                op_features: list=None,
                response_features: list=None,
                op_doc_features: list=None,
                response_doc_features: list=None,                                
                ) -> Dict[str, torch.Tensor]:

        real_output = self._cmv_predictor(response, label, original_post, op_features=op_features, response_features=response_features, op_doc_features=op_doc_features, response_doc_features=response_doc_features)

        document_to_compress = original_post
        if self._compress_response:
            document_to_compress = response
        document, mask = self._document_embedder(response)
        idxs, _, _ = self._cmv_extractor(document, mask, label)
        fake_output = self._cmv_predictor(response, label, original_post, fake_data=True, idxs=idxs, compress_response=self._compress_response,
                                          op_features=op_features, response_features=response_features, op_doc_features=op_doc_features, response_doc_features=response_doc_features)
        
        #CHANGED
        #extracted_sentences = extract(original_post, idxs)
        #fake_output = self._cmv_predictor(response, label, extracted_sentences, fake_data=True)
        
        for param in self._cmv_discriminator.parameters():
            param.requires_grad = True

        #TODO: actually i think we want to detach the reps here
        d_loss = self._cmv_discriminator(real_output['representation'].detach(),
                                         fake_output['representation'].detach())
                
        return {'loss': d_loss['loss']}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        predictor_metrics = self._cmv_predictor.get_metrics(reset)
        discriminator_metrics = self._cmv_discriminator.get_metrics(reset)

        metrics = {'P' + key:value for key,value in predictor_metrics.items()}
        metrics.update({'D' + key:value for key,value in discriminator_metrics.items()})

        return metrics
        
                
