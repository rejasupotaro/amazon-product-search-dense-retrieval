from amazon_product_search_dense_retrieval.encoders import BERTEncoder


def test_encode():
    num_proj = 128
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    vectors = encoder.encode(["ナイキの靴"])
    assert vectors.shape == (1, num_proj)


def test_encode_many_texts():
    num_proj = 128
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    vectors = encoder.encode(["text1", "text2", "text3"], batch_size=2)
    assert vectors.shape == (3, num_proj)
