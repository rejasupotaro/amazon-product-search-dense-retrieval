from amazon_product_search_dense_retrieval.encoders import BERTEncoder


# def test_encode():
#     encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese")
#     vectors = encoder.encode(["ナイキの靴"])
#     assert vectors.shape == (1, 128)


def test_encode_many_texts():
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese")
    vectors = encoder.encode(["text1", "text2", "text3"], batch_size=2)
    assert vectors.shape == (3, 128)
