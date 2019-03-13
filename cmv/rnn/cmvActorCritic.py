import itertools

from typing import Dict
import copy

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from allennlp.models import Model
from cmv.rnn.ptr_extractor import LSTMPointerNet

INI = 1e-2

@Model.register("cmv_actor_critic")
class CMVActorCritic(Model):
    """ to be used as critic (predicts a scalar baseline reward)"""    
    def __init__(self,
                 hidden_dim: int,
                 ptr_net: LSTMPointerNet,
                 gamma: float=0.99,
                 dropout: float=0.5,
                 maximize_accuracy: bool=True):
        
        super().__init__(vocab=None)
        
        self._ptr_net = ptr_net
            
        # regression layer
        self._score_linear = nn.Linear(hidden_dim, 1)

        self._gamma = gamma

        self._total_critic_loss = 0
        self._total_rl_loss = 0
        self._total_reward = 0
        self._total_examples = 0

        self._maximize_accuracy = maximize_accuracy
                        
    def forward(self, attn_mem, memory_mask, idxs, prob, labels, predicted, max_step=5):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._ptr_net._prepare(attn_mem)
        scores = []
        lstm_in = lstm_in.transpose(0,1)
        
        for _ in range(max_step):
            output, lstm_states = self._ptr_net._lstm(lstm_in, lstm_states)
            query = output.transpose(0, 1)

            for _ in range(self._ptr_net._n_hop):
                query = LSTMPointerNet.attention(hop_feat, query,
                                            self._ptr_net._hop_v,
                                            self._ptr_net._hop_wq,
                                            None,
                                            mask=memory_mask)
            output = LSTMPointerNet.attention(
                attn_mem, query, self._ptr_net._attn_v,
                self._ptr_net._attn_wq, None, mask=memory_mask,
                attn_feat=attn_feat)
            
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output.transpose(0,1)

        #TODO: stack and reshape scores to be B x MS instead of MS x B            
        scores = torch.stack(scores, dim=0).transpose(0,1)

        #TODO: compute compression_mask from ptr_net
        #mask scores based on compression_mask
        #OR since these have PAD_CHAR maybe dont need to
        
        if self._maximize_accuracy:
            true_weight = 1 if ((labels==1).sum().float() == 0) else ((labels==0).sum().float() / (labels==1).sum().float())
            correct = (labels.byte() == predicted).float()        
            cmv_reward = labels.eq(0).float() * correct + labels.eq(1).float() * correct * true_weight
        else:
            true_positives = 2 * (predicted * labels.eq(1)).float()
            false_positives = -1 * (predicted * labels.eq(0)).float()
            false_negatives = -1 * (predicted.eq(0) * labels.byte()).float()
            cmv_reward = true_positives + false_positives + false_negatives
            
        #compute discounted reward
        indices = []
        probs = []
        rewards = []        
        baselines = []
        #print(fever_reward[0])
        avg_reward = 0
        for i in range(scores.size(0)):
            avg_reward += float(cmv_reward[i])
            length = idxs[i].ne(LSTMPointerNet.PAD_CHAR).sum()
            rewards.append(self._gamma ** torch.range(length-1,0,-1) * float(cmv_reward[i]))
            #CHANGED rewards.append(self._gamma ** torch.range(0,length) * float(cmv_reward[i]))
            indices.append(idxs[i][:length])
            probs.append(prob[i][:length])
            baselines.append(scores[i][:length])
            
        #print(rewards[0])

        indices = list(itertools.chain(*indices))
        probs = list(itertools.chain(*probs))
        baselines = list(itertools.chain(*baselines))
        baseline = torch.cat(baselines).squeeze()
        
        reward = torch.autograd.Variable(torch.cat(rewards), requires_grad=False)        
        if torch.cuda.is_available() and idxs.is_cuda:
            idx = idxs.get_device()
            reward = reward.cuda(idx)

        # standardize rewards
        reward = (reward - reward.mean()) / (
            reward.std() + float(np.finfo(np.float32).eps))

        #print(reward)
        if self.training:
            avg_advantage = 0
            losses = []
            for action, p, r, b in zip(indices, probs, reward, baseline):
                #print(action, p, r, b)
                action = torch.autograd.Variable(torch.LongTensor([action]))
                if torch.cuda.is_available() and r.is_cuda:
                    idx = r.get_device()
                    action = action.cuda(idx)

                advantage = r - b
                #print(r, b, advantage)
                avg_advantage += advantage
                losses.append(-p.log_prob(action)
                              * (advantage/len(indices))) # divide by T*B
                #print(losses[-1])


            rl_loss = sum(losses)
            self._total_rl_loss += rl_loss.item()
                        
        critic_loss = F.mse_loss(baseline, reward)        
        self._total_critic_loss += critic_loss.item()
        self._total_reward += avg_reward
        self._total_examples += labels.size(0)  
        
        return {'loss': critic_loss + rl_loss if self.training else critic_loss,
                'critic_loss': critic_loss.data[0],
                'rl_loss': rl_loss.data[0] if self.training else 0,
                'reward': avg_reward}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict(critic=self._total_critic_loss/self._total_examples,
                       rl=self._total_rl_loss/self._total_examples,
                       reward=self._total_reward/self._total_examples)
        if reset:
            self._total_critic_loss = 0
            self._total_rl_loss = 0
            self._total_reward = 0
            self._total_examples = 0
            
        return metrics
