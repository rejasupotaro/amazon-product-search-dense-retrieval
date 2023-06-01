import torch
from torch import Tensor
from torch.nn import Module, functional


class ContrastiveLoss(Module):
    def __init__(self, margin: float = 0.4):
        super().__init__()
        self.margin = margin

    def forward(self, query: Tensor, pos_doc: Tensor, neg_doc: Tensor) -> Tensor:
        pos_logit = functional.cosine_similarity(query, pos_doc)
        neg_logit = functional.cosine_similarity(query, neg_doc)
        loss = torch.clamp(neg_logit - pos_logit + self.margin, min=0)
        return loss


class InBatchContrastiveLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, pos_doc: Tensor, neg_doc: Tensor) -> Tensor:
        pos_logit = (query * pos_doc).sum(-1).unsqueeze(1)
        neg_logit = torch.matmul(query, neg_doc.transpose(0, 1))
        logit = torch.cat([pos_logit, neg_logit], dim=1)
        lsm = functional.log_softmax(logit, dim=1)
        loss = -1.0 * lsm[:,0]
        return loss
