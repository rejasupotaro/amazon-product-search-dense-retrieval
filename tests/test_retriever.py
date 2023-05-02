import numpy as np
import pytest

from amazon_product_search_dense_retrieval.retriever import Retriever


def test_retrieve():
    doc_ids = ["1", "2"]
    doc_embs_list = [
        np.array([[0.5, 0.5], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [0.5, 0.5]]),
    ]
    weights = [0.6, 0.4]
    retriever = Retriever(
        dim=2,
        doc_ids=doc_ids,
        doc_embs_list=doc_embs_list,
        weights=weights,
    )

    query = np.array([1.0, 1.0])
    doc_ids, scores = retriever.retrieve(query, top_k=2)

    assert doc_ids == ["2", "1"]
    assert scores == [1.6, 1.4]


def test_with_inconsistent_indices():
    doc_ids = ["1", "2"]
    doc_embs_list = [
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    ]
    weights = [1, 1]
    retriever = Retriever(
        dim=2,
        doc_ids=doc_ids,
        doc_embs_list=doc_embs_list,
        weights=weights,
    )

    query = np.array([1.0, 1.0])
    with pytest.raises(AssertionError):
        retriever.retrieve(query, top_k=1)
