from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric

class ExtractorScore(Metric):
    def __init__(self) -> None:
        self.macro_precision = 0.
        self.macro_precision_hits = 0.
        self.macro_recall = 0.
        self.macro_recall_hits = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 labels: torch.Tensor,
                 pad_idx=-1):

        for i in range(predictions.size(0)):
            if int(labels[i].ne(pad_idx).sum()) == 0:
                continue

            l = set(labels[i].masked_select(labels[i].ne(pad_idx)).data.cpu().numpy())
            p = set(predictions[i].masked_select(predictions[i].ne(pad_idx)).data.cpu().numpy())

            found = len(p & l) * 1.
            #print(l, p, (found / len(p)) if len(p) else 1.0, found / len(l))
            
            self.macro_precision += (found / len(p)) if len(p) else 1.0
            self.macro_recall += found / len(l)
            
            self.macro_precision_hits += 1
            self.macro_recall_hits += 1

    def get_metric(self, reset: bool = False):
            """
            Returns
            -------
            The accumulated accuracy.
            """


            #print(self.macro_precision, self.macro_precision_hits, self.macro_recall, self.macro_recall_hits)
            
            pr = (self.macro_precision / self.macro_precision_hits) if self.macro_precision_hits > 0 else 1.0
            rec = (self.macro_recall / self.macro_recall_hits) if self.macro_recall_hits > 0 else 0.0

            f1 = 0
            if pr + rec:
                f1 = 2.0 * pr * rec / (pr + rec)

            if reset:
                self.reset()
            return pr, rec, f1

    @overrides
    def reset(self):
        self.macro_precision = 0.
        self.macro_precision_hits = 0.
        self.macro_recall = 0.
        self.macro_recall_hits = 0.        
