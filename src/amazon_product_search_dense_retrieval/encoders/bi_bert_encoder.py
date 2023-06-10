from torch import Tensor
from torch.nn import Module, functional

from amazon_product_search_dense_retrieval.encoders.bert_encoder import (
    BERTEncoder,
    RepMode,
)


class BiBERTEncoder(Module):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool,
        rep_mode: RepMode,
        projection_shape: tuple[int, int],
        criteria: Module,
    ) -> None:
        super().__init__()
        self.encoder = BERTEncoder(
            bert_model_name,
            bert_model_trainable,
            rep_mode,
            projection_shape,
        )
        self.criteria = criteria

    @staticmethod
    def compute_score(query: Tensor, doc: Tensor) -> Tensor:
        return functional.cosine_similarity(query, doc)

    def forward(
        self,
        query: dict[str, Tensor],
        pos_doc: dict[str, Tensor],
        neg_doc: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        query_vec = self.encoder.forward(query, target="query")
        pos_doc_vec = self.encoder.forward(pos_doc, target="doc")
        neg_doc_vec = self.encoder.forward(neg_doc, target="doc")
        loss: Tensor = self.criteria(query_vec, pos_doc_vec, neg_doc_vec)
        pos_score = self.compute_score(query_vec, pos_doc_vec)
        neg_score = self.compute_score(query_vec, neg_doc_vec)
        acc = (pos_score > neg_score).float()
        return (loss.mean(), acc.mean())
