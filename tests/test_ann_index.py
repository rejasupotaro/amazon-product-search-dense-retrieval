import numpy as np

from amazon_product_search_dense_retrieval.ann_index import ANNIndex


def test_search():
    index = ANNIndex(dim=2)
    doc_ids = np.array(["1", "2", "3"])
    doc_vectors = np.array([[-1, -1], [0, 0], [1, 1]])
    index.reset(doc_ids, doc_vectors)

    query = np.array([0, 0])
    doc_ids, scores = index.search(query, 1)
    assert doc_ids == ["1"]
    assert scores == [0.0]


def test_save_and_load(tmp_path):
    index = ANNIndex(dim=2)
    doc_ids = np.array(["1", "2", "3"])
    doc_vectors = np.array([[-1, -1], [0, 0], [1, 1]])
    index.reset(doc_ids, doc_vectors)

    index_filepath = str(tmp_path / "test")
    index.save(index_filepath)

    index = ANNIndex(dim=2)
    index.load(index_filepath)

    query = np.array([0, 0])
    doc_ids, scores = index.search(query, 1)

    assert doc_ids == ["1"]
    assert scores == [0.0]
