import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cosine_similarity, log_softmax, normalize


class TripletLoss(Module):
    def __init__(self, margin: float = 0.4) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, query: Tensor, pos_doc: Tensor, neg_doc: Tensor) -> Tensor:
        pos_logit = cosine_similarity(query, pos_doc)
        neg_logit = cosine_similarity(query, neg_doc)
        loss = torch.clamp(neg_logit - pos_logit + self.margin, min=0)
        return loss


class InfoNCELoss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: Tensor, pos_doc: Tensor, neg_doc: Tensor) -> Tensor:
        # Normalize the given vectors before computing the doc product.
        query = normalize(query, dim=-1)
        pos_doc = normalize(pos_doc, dim=-1)
        neg_doc = normalize(neg_doc, dim=-1)

        # Compute the dot product of
        # 1. query (batch_size, vec_len) and pos_doc (batch_size, vec_len) => (batch_size) => (batch_size, 1)
        pos_logit = (query * pos_doc).sum(-1).unsqueeze(-1)
        # 1. query (batch_size, vec_len) and neg_docs (batch_size, vec_len).T => (batch_size, batch_size)
        neg_logit = torch.matmul(query, neg_doc.t())

        # Concat pos_logit and neg_logits => (batch_size, batch_size + 1)
        logit = torch.cat([pos_logit, neg_logit], dim=-1)
        lsm = log_softmax(logit, dim=-1)
        loss = -1.0 * lsm[:, 0]
        return loss
