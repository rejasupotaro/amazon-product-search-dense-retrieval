import numpy as np

from amazon_product_search_dense_retrieval.ann_index import ANNIndex


def test_search():
    index = ANNIndex(dim=2)
    doc_ids = np.array(["1", "2", "3"])
    doc_embs = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    index.reset(doc_ids, doc_embs)

    query = np.array([1.0, 1.0])
    doc_ids, scores = index.search(query, 2)
    assert doc_ids == ["3", "2"]
    assert scores == [2.0, 0.0]


def test_save_and_load(tmp_path):
    index = ANNIndex(dim=2)
    doc_ids = np.array(["1", "2", "3"])
    doc_embs = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    index.reset(doc_ids, doc_embs)

    index_filepath = str(tmp_path / "test")
    index.save(index_filepath)

    index = ANNIndex(dim=2)
    index.load(index_filepath)

    query = np.array([1.0, 1.0])
    doc_ids, scores = index.search(query, 1)

    assert doc_ids == ["3"]
    assert scores == [2.0]
