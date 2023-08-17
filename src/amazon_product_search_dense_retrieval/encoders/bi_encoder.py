from torch import Tensor
from torch.nn import Module, functional

from amazon_product_search_dense_retrieval.encoders.text_encoder import TextEncoder


class BiEncoder(Module):
    def __init__(
        self,
        query_encoder: TextEncoder,
        product_encoder: TextEncoder,
        criteria: Module,
    ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.product_encoder = product_encoder
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
        query_vec = self.query_encoder(query)
        pos_doc_vec = self.product_encoder(pos_doc)
        neg_doc_vec = self.product_encoder(neg_doc)
        loss: Tensor = self.criteria(query_vec, pos_doc_vec, neg_doc_vec)
        pos_score = self.compute_score(query_vec, pos_doc_vec)
        neg_score = self.compute_score(query_vec, neg_doc_vec)
        acc = (pos_score > neg_score).float()
        return (loss.mean(), acc.mean())
