import pytest
import torch

from amazon_product_search_dense_retrieval.encoders.modules.pooler import Pooler


@pytest.mark.parametrize(
    ("pooling_mode", "expected"),
    [
        ("cls", [[0.5, 0.5]]),
        ("mean", [[0.5, 0.5]]),
        ("max", [[1.0, 1.0]]),
    ],
)
def test_forward(pooling_mode, expected):
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
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    pooler = Pooler(pooling_mode)
    text_emb = pooler.forward(token_embs, attention_mask)
    assert text_emb.tolist() == expected
