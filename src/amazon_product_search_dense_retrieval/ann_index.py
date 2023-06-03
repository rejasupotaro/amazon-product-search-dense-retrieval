import pickle

import numpy as np
from annoy import AnnoyIndex


class ANNIndex:
    def __init__(self, dim: int):
        self.annoy_index = AnnoyIndex(dim, "dot")
        self._doc_id_to_idx: dict[int, str] = {}

    def reset(self, doc_ids: list[str], doc_embs: np.ndarray):
        for idx, (doc_id, doc_emb) in enumerate(zip(doc_ids, doc_embs, strict=True)):
            self._doc_id_to_idx[idx] = doc_id
            self.annoy_index.add_item(idx, doc_emb)
        self.annoy_index.build(10)

    def search(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        doc_ids = []
        scores = []
        retrieved = self.annoy_index.get_nns_by_vector(
            query, top_k, include_distances=True
        )
        for idx, score in zip(*retrieved, strict=True):
            doc_ids.append(self._doc_id_to_idx[idx])
            scores.append(score)
        return doc_ids, scores

    def save(self, index_filepath: str):
        self.annoy_index.save(f"{index_filepath}.ann")
        with open(f"{index_filepath}.pkl", "wb") as file:
            pickle.dump(self._doc_id_to_idx, file)

    def load(self, index_filepath: str):
        self.annoy_index.load(f"{index_filepath}.ann")
        with open(f"{index_filepath}.pkl", "rb") as file:
            self._doc_id_to_idx = pickle.load(file)
