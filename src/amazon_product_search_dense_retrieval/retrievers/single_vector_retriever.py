import numpy as np

from ..ann_index import ANNIndex


class SingleVectorRetriever:
    def __init__(self, dim: int, doc_ids: list[str], doc_embs: np.ndarray):
        self.ann_index = ANNIndex(dim=dim)
        self.ann_index.reset(doc_ids, doc_embs)

    def retrieve(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        return self.ann_index.search(query, top_k)
