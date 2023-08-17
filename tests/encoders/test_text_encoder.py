import pytest

from amazon_product_search_dense_retrieval.encoders.text_encoder import (
    TextEncoder,
    load,
    save,
)


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        ("text", (768,)),
        (["text"], (1, 768)),
    ],
)
def test_encode(texts, expected):
    encoder = TextEncoder(
        hf_model_name="sonoisa/sentence-luke-japanese-base-lite",
        hf_model_trainable=False,
        pooling_mode="cls",
    )

    text_embs = encoder.encode(texts)
    assert text_embs.shape == expected


def test_save_and_load(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    models_dir = str(models_dir)
    hf_model_name = "sonoisa/sentence-luke-japanese-base-lite"
    hf_model_trainable = False
    pooling_mode = "cls"

    encoder = TextEncoder(
        hf_model_name="sonoisa/sentence-luke-japanese-base-lite",
        hf_model_trainable=False,
        pooling_mode=pooling_mode,
    )
    model_name = "text_encoder"
    model_filepath = save(models_dir, model_name, encoder)
    assert model_filepath.split("/")[-1] == "text_encoder.pt"

    encoder = load(
        models_dir, model_name, hf_model_name, hf_model_trainable, pooling_mode
    )
