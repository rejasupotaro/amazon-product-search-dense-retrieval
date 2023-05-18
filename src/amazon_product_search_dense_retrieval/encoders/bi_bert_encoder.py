import torch
from torch import Tensor
from torch.nn import Module, functional

from amazon_product_search_dense_retrieval.encoders.bert_encoder import BERTEncoder, RepMode


class BiBERTEncoder(Module):
    def __init__(self, bert_model_name: str, bert_model_trainable: bool, rep_mode: RepMode, num_proj: int):
        super().__init__()
        self.encoder = BERTEncoder(bert_model_name, bert_model_trainable, rep_mode, num_proj)

    @staticmethod
    def in_batch_contrastive_loss(query: Tensor, pos_doc: Tensor, neg_doc: Tensor) -> Tensor:
        pos_logit = (query * pos_doc).sum(-1).unsqueeze(1)
        neg_logit = torch.matmul(query, neg_doc.transpose(0, 1))
        logit = torch.cat([pos_logit, neg_logit], dim=1)
        lsm = functional.log_softmax(logit, dim=1)
        loss = -1.0 * lsm[:,0]
        return loss

    def forward(
        self, query: dict[str, Tensor], pos_doc: dict[str, Tensor], neg_doc: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        query_vec, pos_doc_vec, neg_doc_vec = self.encoder(query), self.encoder(pos_doc), self.encoder(neg_doc)
        loss = self.in_batch_contrastive_loss(query_vec, pos_doc_vec, neg_doc_vec)
        pos_score = (query_vec * pos_doc_vec).sum(dim=1)
        neg_score = (query_vec * neg_doc_vec).sum(dim=1)
        acc = (pos_score > neg_score).float()
        return (loss.mean(), acc.mean())
