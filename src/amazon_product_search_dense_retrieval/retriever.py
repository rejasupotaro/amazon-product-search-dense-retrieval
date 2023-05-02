from collections import defaultdict

import numpy as np

from .ann_index import ANNIndex


class Retriever:
    def __init__(self, dim: int, doc_ids: list[str], doc_embs_list: list[np.ndarray], weights: list[float]):
        self.ann_indices: list[ANNIndex] = []
        for doc_embs in doc_embs_list:
            ann_index = ANNIndex(dim=dim)
            ann_index.reset(doc_ids, doc_embs)
            self.ann_indices.append(ann_index)
        self.weights = weights

    def retrieve(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        assert len(self.ann_indices) == len(self.weights)

        candidates: dict[str, float] = defaultdict(float)
        for ann_index, weight in zip(self.ann_indices, self.weights):
            doc_ids, scores = ann_index.search(query, top_k)
            for doc_id, score in zip(doc_ids, scores):
                candidates[doc_id] += score * weight
        sorted_candidates = sorted(candidates.items(), key=lambda id_and_score: id_and_score[1], reverse=True)

        doc_ids = []
        scores = []
        for doc_id, score in sorted_candidates[:top_k]:
            doc_ids.append(doc_id)
            scores.append(score)
        return doc_ids, scores
