#this module "generates" new original posts by extracting the sentences/components that are most important for predicting persuasion
#for now, we just do extractive summarization with a pointer network and bridge the gap with reinforcement learning
#TODO: use Gumbel softmax or something instead of discrete selection
#TODO: fill in gaps between extracted sentences with abstraction

#the loss for the generator is -L_D(fake_data) + L_P(fake_data) + L_P(real_data)

#maybe + L_ptr (using quoted text as distant supervision)
#maybe + L_pop(fake_data) (predicting persuasion from extracted sentences only)

#TODO: separate pointer networks for persuasive and non-persuasive (this might cause an issue with the discriminator but maybe not if we use REINFORCE)

#reward could also be interplay, or L_ptr

from typing import Dict

import torch

from allennlp.models import Model

from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvExtractor import CMVExtractor, extract
from cmv.rnn.cmvDiscriminator import CMVDiscriminator
from cmv.rnn.cmvActorCritic import CMVActorCritic

@Model.register("cmv_generator_trainer")
class CMVGeneratorTrainer(Model):
    def __init__(self,
                 document_embedder: HierarchicalDocumentEmbedder,
                 cmv_predictor: CMVPredictor,
                 cmv_extractor: CMVExtractor,
                 cmv_discriminator: CMVDiscriminator,
                 cmv_actor_critic: CMVActorCritic,
                 update_extractor: bool=True,
                 update_gold_extractor: bool=False,
                 compress_response: bool=False):

        super().__init__(vocab=None)

        self._document_embedder = document_embedder
        
        self._cmv_predictor = cmv_predictor
        self._cmv_extractor = cmv_extractor
        self._cmv_discriminator = cmv_discriminator
        self._cmv_actor_critic = cmv_actor_critic

        self._update_extractor = update_extractor
        self._update_gold_extractor = update_gold_extractor

        assert(update_extractor or update_gold_extractor)

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
            weakpoints=None
            document_to_compress = response
            
        document, mask = self._document_embedder(document_to_compress)
        
        if self._update_gold_extractor:
            #first get the loss using teacher forcing
            save_train_rl = self._cmv_extractor._train_rl
            self._cmv_extractor._train_rl = False
            idxs, _, gold_loss = self._cmv_extractor(document, mask, label,
                                                        gold_evidence=weakpoints)
            self._cmv_extractor._train_rl = save_train_rl
            #print(idxs)
            #then get the model predictions for all the data, including the ones without quotes
            idxs, _, _ = self._cmv_extractor(document, mask, label,
                                                        teacher_forcing=False)
            #print(idxs)
        else:
            idxs, probs, gold_loss = self._cmv_extractor(document, mask, label,
                                                        gold_evidence=weakpoints)
            
        #extracted_sentences = extract(original_post, idxs)

        #TODO: do we want this? should this be a separate predictor?        
        fake_output = self._cmv_predictor(response, label, original_post, idxs=idxs, #extracted_sentences,
                                          fake_data=True,
                                          op_features=op_features, response_features=response_features,
                                          compress_response=self._compress_response, op_doc_features=op_doc_features, response_doc_features=response_doc_features)

        for param in self._cmv_discriminator.parameters():
            param.requires_grad = False
        d_loss = self._cmv_discriminator(fake_output['representation'])
        
        if self._update_extractor:
            if self._update_gold_extractor:
                idxs, probs, _ = self._cmv_extractor(document, mask, label,
                                                            gold_evidence=weakpoints)
                #print(idxs)
                #extracted_sentences = extract(original_post, idxs)

                fake_output = self._cmv_predictor(response, label, original_post, idxs=idxs,
                                                  op_features=op_features, response_features=response_features,
                                                  fake_data=True, compress_response=self._compress_response, op_doc_features=op_doc_features, response_doc_features=response_doc_features)
                
            rl_loss = self._cmv_actor_critic(document,
                                            mask,
                                            idxs,
                                            probs,
                                            label,
                                            fake_output['label_probs'] > 0.5)
                        
        persuasion_loss = real_output['loss'] #+ fake_output['loss']

        loss = d_loss['loss'] + persuasion_loss + gold_loss
        if self._update_extractor:
            return {'loss': loss + rl_loss['loss']}
        return {'loss': loss}
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        predictor_metrics = self._cmv_predictor.get_metrics(reset)
        discriminator_metrics = self._cmv_discriminator.get_metrics(reset)
        if self._update_extractor:        
            actor_critic_metrics = self._cmv_actor_critic.get_metrics(reset)
        else:
            actor_critic_metrics = {}
            
        metrics = {'P' + key:value for key,value in predictor_metrics.items()}
        metrics.update({'D' + key:value for key,value in discriminator_metrics.items()})
        metrics.update({'AC' + key:value for key,value in actor_critic_metrics.items()})    

        extractor_metrics = self._cmv_extractor.get_metrics(reset)        
        if extractor_metrics['recall'] > 0:
            metrics.update({'EX' + key:value for key,value in extractor_metrics.items()})    
        
        return metrics
