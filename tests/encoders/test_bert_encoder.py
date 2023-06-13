import pytest

from amazon_product_search_dense_retrieval.encoders import BERTEncoder


@pytest.mark.parametrize(
    ("texts", "projection_shape", "expected"),
    [
        ("text", (768, 128), (128,)),
        (["text"], (768, 128), (1, 128)),
    ],
)
def test_encode(texts, projection_shape, expected):
    encoder = BERTEncoder(
        bert_model_name="sonoisa/sentence-luke-japanese-base-lite",
        bert_model_trainable=False,
        rep_mode="cls",
        projection_shape=projection_shape,
    )

    text_embs = encoder.encode(texts, target="query")
    assert text_embs.shape == expected


def test_encode_many_texts():
    projection_shape = (768, 4)
    encoder = BERTEncoder(
        bert_model_name="sonoisa/sentence-luke-japanese-base-lite",
        bert_model_trainable=False,
        rep_mode="cls",
        projection_shape=projection_shape,
    )

    text_embs = encoder.encode(
        ["text1", "text2", "text3"], target="query", batch_size=2
    )
    assert text_embs.shape == (3, 4)

    text_embs = encoder.encode(["text1", "text2", "text3"], target="doc", batch_size=2)
    assert text_embs.shape == (3, 768)


def test_save_and_load(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    models_dir = str(models_dir)
    bert_model_name = "sonoisa/sentence-luke-japanese-base-lite"
    bert_model_trainable = False
    rep_mode = "mean"
    projection_shape = (768, 768)

    encoder = BERTEncoder(
        bert_model_name,
        bert_model_trainable,
        rep_mode,
        projection_shape,
    )
    model_filepath = encoder.save(models_dir)
    assert (
        model_filepath.split("/")[-1]
        == "sonoisa_sentence-luke-japanese-base-lite_mean_768_768.pt"
    )

    encoder = BERTEncoder.load(
        bert_model_name,
        bert_model_trainable,
        rep_mode,
        projection_shape,
        models_dir,
    )
