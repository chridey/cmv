#train the extractor to maximize the accuracy on the predictor
#don't train the predictor at this time
#then later, fix the extractor and train only the representations

from typing import Dict, Optional

import torch

from allennlp.models import Model

from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvExtractor import CMVExtractor, extract
from cmv.rnn.cmvActorCritic import CMVActorCritic

@Model.register("cmv_actor_critic_trainer")
class CMVActorCriticTrainer(Model):
    def __init__(self,
                 document_embedder: HierarchicalDocumentEmbedder,
                 cmv_predictor: CMVPredictor,                 
                 cmv_extractor: CMVExtractor,
                 cmv_actor_critic: Optional[CMVActorCritic] = None,
                 train_predictor: bool=False,
                 train_fake_predictor: bool=False,
                 compress_response: bool=False):

        super().__init__(vocab=None)
        
        self._document_embedder = document_embedder
        
        self._cmv_predictor = cmv_predictor
        self._cmv_extractor = cmv_extractor
        self._cmv_actor_critic = cmv_actor_critic

        self._train_predictor = train_predictor
        self._train_fake_predictor = train_fake_predictor        

        self._compress_response = compress_response
        
    def forward(self,
                response: Dict[str, torch.LongTensor],
                label: torch.IntTensor,
                original_post: Dict[str, torch.LongTensor],
                weakpoints: Optional[torch.IntTensor] = None,
                op_features: list=None,
                response_features: list=None,
                op_doc_features: list=None,
                response_doc_features: list=None,                
                ) -> Dict[str, torch.Tensor]:

        document_to_compress = original_post
        if self._compress_response:
            document_to_compress = response
            weakpoints=None
            
        document, mask = self._document_embedder(document_to_compress)
        print(document.shape, mask.shape)
        idxs, probs, gold_loss = self._cmv_extractor(document, mask, label,
                                                gold_evidence=weakpoints)
        print(idxs)
        
        loss = 0
        if gold_loss is not None:
            loss = gold_loss

        if self._train_predictor:
            output = self._cmv_predictor(response, label, original_post, op_features=op_features, response_features=response_features, op_doc_features=op_doc_features, response_doc_features=response_doc_features)
            loss += output['loss']

        #print(idxs.shape, weakpoints.squeeze(-1).shape,
        #      response['tokens'].shape, original_post['tokens'].shape)
        if self._train_fake_predictor or self._cmv_actor_critic is not None:
            #extracted_sentences = extract(original_post, idxs)
            #TODO: teacher forcing (gold evidence) or predicted
            #TODO: doesnt work when weakpoints are -1
            fake_output = self._cmv_predictor(response, label, original_post,
                                              idxs=idxs, fake_data=True, compress_response=self._compress_response,
                                              op_features=op_features, response_features=response_features, op_doc_features=op_doc_features, response_doc_features=response_doc_features)

        if self._train_fake_predictor:
            loss += fake_output['loss']
                            
        if self._cmv_actor_critic is not None:

            rl_loss = self._cmv_actor_critic(document,
                                            mask,
                                            idxs,
                                            probs,
                                            label,
                                            fake_output['label_probs'] > 0.5)
            loss += rl_loss['loss']
            
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if self._train_predictor or self._cmv_actor_critic is not None:
            predictor_metrics = self._cmv_predictor.get_metrics(reset)                
            metrics.update({'P' + key:value for key,value in predictor_metrics.items()})

        if self._cmv_actor_critic is not None:
            actor_critic_metrics = self._cmv_actor_critic.get_metrics(reset)        
            metrics.update({'AC' + key:value for key,value in actor_critic_metrics.items()})    

        extractor_metrics = self._cmv_extractor.get_metrics(reset)        
        if extractor_metrics['recall'] > 0:
            metrics.update({'EX' + key:value for key,value in extractor_metrics.items()})    
        
        return metrics
    
