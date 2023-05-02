from collections import defaultdict

import numpy as np

from .ann_index import ANNIndex


class Retriever:
    """A retriever that finds the most relevant documents to a given query.

    This class is just for training and evaluation. In practice, you should use a distributed search engine
    that is capable to handle large-scale documents.

    Args:
        dim (int): The dimensionality of the document embeddings.
        doc_ids (list[str]): A list of document IDs, where each ID corresponds to an embedding in `doc_embs_list`.
        doc_embs_list (list[np.ndarray]): A list of document embeddings, where each embedding has dimension `dim`.
        weights (list[float]): A list of weights, where each weight corresponds to a set of embeddings.

    Attributes:
        weights (list[float]): A list of weights, where each weight corresponds to a set of embeddings.
    """

    def __init__(self, dim: int, doc_ids: list[str], doc_embs_list: list[np.ndarray], weights: list[float]):
        self._ann_indices: list[ANNIndex] = []
        for doc_embs in doc_embs_list:
            ann_index = ANNIndex(dim=dim)
            ann_index.reset(doc_ids, doc_embs)
            self._ann_indices.append(ann_index)
        self.weights = weights

    def retrieve(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        """Retrieves the top `k` most relevant documents to a given query.

        This function searches each ANNIndex object for the `k` most similar document embeddings to the query,
        and computes a score for each document based on the weighted sum of its scores across all ANNIndex objects.
        The resulting documents are returned in order of decreasing score.

        Args:
            query (np.ndarray): A numpy array representing the query vector, with shape (dim,).
            top_k (int): The maximum number of documents to retrieve.

        Returns:
            A tuple of two lists: `doc_ids` and `scores`. The `doc_ids` list contains the IDs of the top `k` documents,
            and the `scores` list contains their corresponding scores.
        """
        assert len(self._ann_indices) == len(self.weights)

        candidates: dict[str, float] = defaultdict(float)
        for ann_index, weight in zip(self._ann_indices, self.weights):
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
