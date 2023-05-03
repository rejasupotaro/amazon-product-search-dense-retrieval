import pytest
import torch

from amazon_product_search_dense_retrieval.encoders import BERTEncoder


@pytest.mark.parametrize(
    "rep_mode,expected",
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

    text_embs = BERTEncoder.convert_token_embs_to_text_emb(token_embs, attention_mask, rep_mode)
    assert text_embs.tolist() == expected


@pytest.mark.parametrize(
    "texts,num_proj,expected",
    [
        ("text", 128, (128,)),
        (["text"], 128, (1,128)),
    ]
)
def test_encode(texts, num_proj, expected):
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    text_embs = encoder.encode(texts)
    assert text_embs.shape == expected


def test_encode_many_texts():
    num_proj = 128
    encoder = BERTEncoder("ku-nlp/deberta-v2-base-japanese", num_proj=num_proj)
    text_embs = encoder.encode(["text1", "text2", "text3"], batch_size=2)
    assert text_embs.shape == (3, num_proj)
