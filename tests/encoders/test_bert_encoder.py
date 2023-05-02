from amazon_product_search_dense_retrieval.encoders import BERTEncoder


def test_encode():
    num_proj = 128
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    text_embs = encoder.encode(["ナイキの靴"])
    assert text_embs.shape == (1, num_proj)


def test_encode_many_texts():
    num_proj = 128
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    text_embs = encoder.encode(["text1", "text2", "text3"], batch_size=2)
    assert text_embs.shape == (3, num_proj)
