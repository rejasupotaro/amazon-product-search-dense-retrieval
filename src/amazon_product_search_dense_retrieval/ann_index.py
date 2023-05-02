from typing import Optional

import faiss
import numpy as np
from faiss import IndexIDMap2


class ANNIndex:
    def __init__(self, faiss_index: Optional[IndexIDMap2] = None, dim: Optional[int] = None):
        if faiss_index:
            self.faiss_index = faiss_index
        elif dim:
            self.faiss_index = IndexIDMap2(faiss.index_factory(dim, "HNSW64", faiss.METRIC_INNER_PRODUCT))
        else:
            raise ValueError("Either faiss_index or dim should be given.")

    @staticmethod
    def load(index_filepath: str) -> "ANNIndex":
        faiss_index = faiss.read_index(index_filepath)
        return ANNIndex(faiss_index=faiss_index)

    def update(self, doc_vectors: np.ndarray, doc_ids):
        self.faiss_index.add_with_ids(doc_vectors, doc_ids)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.array([query])
        scores, doc_ids = self.search_in_batch(queries, top_k)
        return scores[0], doc_ids[0]

    def search_in_batch(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, doc_ids = self.faiss_index.search(queries, top_k)
        return scores, doc_ids

    def save(self, index_filepath: str):
        faiss.write_index(self.faiss_index, index_filepath)
