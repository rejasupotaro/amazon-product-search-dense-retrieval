import numpy as np

from amazon_product_search_dense_retrieval.ann_index import ANNIndex


def test_search():
    index = ANNIndex(dim=2)
    doc_vectors = np.array([[-1, -1], [0, 0], [1, 1]])
    doc_ids = np.array([1, 2, 3])
    index.update(doc_vectors, doc_ids)

    query = np.array([0, 0])
    scores, doc_ids = index.search(query, 1)
    assert scores == np.array([0])
    assert doc_ids == np.array([2])


def test_save_and_load(tmp_path):
    index = ANNIndex(dim=2)
    doc_vectors = np.array([[-1, -1], [0, 0], [1, 1]])
    doc_ids = np.array([1, 2, 3])
    index.update(doc_vectors, doc_ids)

    index_filepath = str(tmp_path / "test.index")
    index.save(index_filepath)
    index = ANNIndex.load(index_filepath)

    query = np.array([0, 0])
    scores, doc_ids = index.search(query, 1)

    assert scores == np.array([0])
    assert doc_ids == np.array([2])
