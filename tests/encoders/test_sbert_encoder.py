from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


def test_encode():
    encoder = SBERTEncoder("ku-nlp/deberta-v2-base-japanese")
    text_embs = encoder.encode(["ナイキの靴"])
    assert text_embs.shape == (1, 768)
