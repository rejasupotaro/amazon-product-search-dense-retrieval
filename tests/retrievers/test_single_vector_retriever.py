import numpy as np

from amazon_product_search_dense_retrieval.retrievers import SingleVectorRetriever


def test_retrieve():
    doc_ids = ["1", "2"]
    doc_embs = np.array([[0.5, 0.5], [1.0, 1.0]])
    retriever = SingleVectorRetriever(
        dim=2,
        doc_ids=doc_ids,
        doc_embs=doc_embs,
    )

    query = np.array([1.0, 1.0])
    doc_ids, scores = retriever.retrieve(query, top_k=2)

    assert doc_ids == ["2", "1"]
    assert scores == [2.0, 1.0]
