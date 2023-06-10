import pytest
import torch

from amazon_product_search_dense_retrieval.encoders import BERTEncoder


@pytest.mark.parametrize(
    ("rep_mode", "expected"),
    [
        ("cls", [[0.5, 0.5]]),
        ("mean", [[0.5, 0.5]]),
        ("max", [[1.0, 1.0]]),
    ],
)
def test_convert_token_embs_to_text_emb(rep_mode, expected):
    token_embs = torch.tensor(
        [
            [
                [0.5, 0.5],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, 0.6],
            ]
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
        ]
    )

    text_embs = BERTEncoder.convert_token_embs_to_text_emb(
        token_embs, attention_mask, rep_mode
    )
    assert text_embs.tolist() == expected


@pytest.mark.parametrize(
    ("texts", "projection_shape", "expected"),
    [
        ("text", (768, 128), (128,)),
        (["text"], (768, 128), (1, 128)),
    ],
)
def test_encode(texts, projection_shape, expected):
    encoder = BERTEncoder(
        "ku-nlp/deberta-v2-base-japanese",
        projection_shape=projection_shape,
    )

    text_embs = encoder.encode(texts, target="query")
    assert text_embs.shape == expected


def test_encode_many_texts():
    projection_shape = (768, 4)
    encoder = BERTEncoder(
        "ku-nlp/deberta-v2-base-japanese",
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
    assert model_filepath.split("/")[-1] == "sonoisa_sentence-luke-japanese-base-lite_mean_768_768.pt"

    encoder = BERTEncoder.load(
        bert_model_name, bert_model_trainable, rep_mode, projection_shape, model_filepath,
    )
