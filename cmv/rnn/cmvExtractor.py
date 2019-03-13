#this module "generates" new original posts by extracting the sentences/components that are most important for predicting persuasion
#for now, we just do extractive summarization with a pointer network and bridge the gap with reinforcement learning
#TODO: use Gumbel softmax or something instead of discrete selection
#TODO: fill in gaps between extracted sentences with abstraction

#the loss for the generator is -L_D(fake_data) + L_P(fake_data) + L_P(real_data)

#maybe + L_ptr (using quoted text as distant supervision)
#maybe + L_pop(fake_data) (predicting persuasion from extracted sentences only)

#TODO: separate pointer networks for persuasive and non-persuasive (this might cause an issue with the discriminator but maybe not if we use REINFORCE)

from typing import Dict

import torch
from torch.nn import functional as F

from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask

from cmv.rnn.ptr_extractor import LSTMPointerNet
from cmv.rnn.extractorMetrics import ExtractorScore

@Model.register("cmv_extractor")
class CMVExtractor(Model):
    """ shared encoder between actor/critic"""
    """ works only on single sample in RL setting"""    
    def __init__(self,
                 ptr_net: LSTMPointerNet,
                 gamma: float=0.99,
                 dropout: float=0.5,
                 use_stop=False,
                 train_rl=True,
                 compression_rate=0):
        
        super().__init__(vocab=None)

        self._ext = ptr_net

        self._extractor_score = ExtractorScore()
                        
        self._use_stop = use_stop
        if use_stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform(self._stop, -INI, INI)

        self._train_rl = train_rl
        self._compression_rate = compression_rate
        
    def _extract(self, attn_mem, n_step, memory_mask=None, gold_evidence=None, k=1,
                 teacher_forcing=True):
        """atten_mem: Tensor of size [num_sents, input_dim]"""

        max_step = attn_mem.size(0)
        if self._use_stop:
            attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
            o = torch.autograd.Variable(torch.ones(1).byte())
            if torch.cuda.is_available() and memory_mask.is_cuda:
                idx = memory_mask.get_device()
                o = o.cuda(idx)
            memory_mask = torch.cat([memory_mask, o],
                                    dim=0)
            
        attn_feat = torch.mm(attn_mem, self._ext._attn_wm)
        hop_feat = torch.mm(attn_mem, self._ext._hop_wm)
        
        outputs = []
        dists = []
        
        lstm_in = self._ext._init_i.view(1,1,-1)
        lstm_states = (self._ext._init_h.unsqueeze(1), self._ext._init_c.unsqueeze(1))
        
        for step in range(n_step):
            output, lstm_states = self._ext._lstm(lstm_in, lstm_states)
            query = output[:, -1, :]
            for hop in range(self._ext._n_hop):
                query = CMVExtractor.attention(hop_feat, query,
                                              self._ext._hop_v, self._ext._hop_wq, mask=memory_mask)
            score = CMVExtractor.attention_score(
                attn_feat, query, self._ext._attn_v, self._ext._attn_wq, mask=memory_mask)
            #print(score.shape)

            #we need to make sure we select at least one element
            if step == 0 and self._use_stop:
                fill = torch.autograd.Variable(torch.LongTensor([max_step]))
                if torch.cuda.is_available() and score.is_cuda:
                    idx = score.get_device()
                    fill = fill.cuda(idx)
                score.index_fill_(1, fill, -1e18)

            #this was moved here to prevent RL training using repeated indices
            if len(outputs):
                score.index_fill_(1, torch.cat(outputs).view(-1), -1e18)

            prob = F.softmax(score, dim=-1)
            #print(prob)
            m = torch.distributions.Categorical(prob)
            dists.append(m)
            
            if self.training and teacher_forcing:
                if gold_evidence is None or self._train_rl:
                    out = m.sample()
                else:
                    out = gold_evidence[step]
                    if out.data[0] < 0:
                        break
            else:
                #print(score.shape)
                out = score.max(dim=1, keepdim=True)[1]

            if out.data[0] == max_step:
                break                    
                
            #print(out, out.shape)
            outputs.append(out)

            #print(out.view(-1).data[0])
            #if out.view(-1).data[0] == max_step:
            #    break
            
            lstm_in = attn_mem[out.data[0]].view(1,1,-1)

        #pad the predictions here
        '''
        if len(outputs) < n_step:
            if torch.cuda.is_available and outputs[-1].is_cuda:
                outputs += [torch.cuda.LongTensor([LSTMPointerNet.PAD_CHAR]) for i in range(n_step-len(outputs))]
            else:
                outputs += [torch.LongTensor([LSTMPointerNet.PAD_CHAR]) for i in range(n_step-len(outputs))]
            dists += [torch.distributions.Categorical(torch.zeros(1)) for i in range(n_step-len(outputs))]
        '''
            
        return torch.cat(outputs, dim=0).view(-1), dists

    @staticmethod
    def attention_score(attention, query, v, w, mask=None):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        
        if mask is None:
            return score
        
        return score + mask.eq(0).float() * -1e8
    
    @staticmethod
    def attention(attention, query, v, w, mask=None):
        """ attention context vector"""
        score = F.softmax(
            CMVExtractor.attention_score(attention, query, v, w, mask=mask), dim=-1)
        output = torch.mm(score, attention)
        return output
        
    def forward(self, enc_out, memory_mask, label=None, gold_evidence=None, n_abs=3,
                beam_size=5, teacher_forcing=True):

        #make sure that this is B x N instead of B x N x 1        
        if gold_evidence is not None and len(gold_evidence.shape) > 2:
            gold_evidence = gold_evidence.squeeze(-1)

        if self.training and not self._train_rl:
            if gold_evidence.ne(LSTMPointerNet.PAD_CHAR).sum() == 0:
                z = torch.zeros_like(memory_mask).long()
                o = torch.ones_like(memory_mask).float()
                l = torch.FloatTensor([0])
                if torch.cuda.is_available() and enc_out.is_cuda:
                    idx = enc_out.get_device()
                    l = l.cuda(idx)
                return z, o, torch.autograd.Variable(l, requires_grad=True).clone()
            
            #print(enc_out.shape, memory_mask.shape)
            bs, nt = gold_evidence.size()
            d = enc_out.size(2)

            ptr_in = torch.gather(
                enc_out, dim=1, index=gold_evidence.clamp(min=0).unsqueeze(2).expand(bs, nt, d)
                )
            #print(ptr_in.shape)
            scores = self._ext(enc_out, ptr_in, memory_mask=memory_mask)
            probs, idxs = F.softmax(scores, dim=-1).max(dim=-1)

            #print(scores)
            #print('SEQUENCE LOSS')
            loss = sequence_loss(scores[:,:-1,:], gold_evidence, pad_idx=LSTMPointerNet.PAD_CHAR)
        else:
            idxs = []
            probs = []
            for i in range(enc_out.size(0)):
                document_length = int(memory_mask[i].sum())
                if self._compression_rate > 0:
                    document_length = int(self._compression_rate * document_length + 1)

                if not self.training and beam_size >= 1:
                    idx = self._ext.extract(enc_out[i].unsqueeze(0), None,
                                            min(n_abs, document_length),
                                            mask=memory_mask[i].unsqueeze(0),
                                            beam_size=beam_size).view(-1)
                    prob = [torch.distributions.Categorical(torch.zeros(1))]

                else:
                    idx, prob = self._extract(enc_out[i], min(n_abs, document_length),
                                              memory_mask=memory_mask[i],
                                              gold_evidence=gold_evidence[i] if gold_evidence is not None else None, teacher_forcing=teacher_forcing)

                idxs.append(idx)
                probs.append(prob)

            #need to pad these here
            for i in range(len(idxs)):
                if idxs[i].size(0) < memory_mask.size(1):
                    l = [torch.LongTensor([LSTMPointerNet.PAD_CHAR]) for i in range(memory_mask.size(1) - idxs[i].size(0))]
                    if torch.cuda.is_available() and enc_out.is_cuda:
                        idx = enc_out.get_device()
                        l = [ll.cuda(idx) for ll in l]
                    idxs[i] = torch.cat([idxs[i]] + l)
                    probs[i] += [torch.distributions.Categorical(torch.zeros(1)) for i in range(memory_mask.size(1) - idxs[i].size(0))]
            
            idxs = torch.stack(idxs, dim=0)
            l = torch.FloatTensor([0])
            if torch.cuda.is_available() and enc_out.is_cuda:
                idx = enc_out.get_device()
                l = l.cuda(idx)
            loss = torch.autograd.Variable(l, requires_grad=True).clone()
            
        if gold_evidence is not None:
            self._extractor_score(idxs, gold_evidence)

        #print(gold_evidence)
        #print(idxs)
                        
        return idxs, probs, loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self._extractor_score.get_metric(reset)
        return {'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}

def sequence_loss(logits, targets, pad_idx=-1):
    """ functional interface of SequenceLoss"""
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    #print(logits, targets, mask)
    target = targets.masked_select(mask)
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)
    ).contiguous().view(-1, logits.size(-1))

    #print(logit, target)

    loss = F.cross_entropy(logit, target)
    #assert (not math.isnan(loss.mean().item())
    #        and not math.isinf(loss.mean().item()))

    #print('LOSS', loss)    
    return loss
                                                            
        
def extract(original_post,
            idxs,
            features=None):

    batch_size, max_select = idxs.shape
    key = list(original_post.keys())[0]
    max_length = original_post[key].size(-1)
            
    index=idxs.unsqueeze(2).expand(batch_size,
                                   max_select,
                                   max_length)
    mask = index.ne(LSTMPointerNet.PAD_CHAR).long()
    index = index * mask

    #print(index[0])
    extracted_sentences = {}
    for key in original_post:
        #print(original_post[key][0])
        extracted_sentences[key] = torch.gather(original_post[key], dim=1, index=index)
        #print(extracted_sentences[key][0])

        #zero out padding
        if 'Float' in extracted_sentences[key].type():
            extracted_sentences[key] *= mask.float()
        else:
            extracted_sentences[key] *= mask

    if features is not None:
        features = torch.gather(features, dim=1, index=index)
        features *= mask
        return features
            
    return extracted_sentences


