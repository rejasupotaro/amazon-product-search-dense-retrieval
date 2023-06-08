import torch

from amazon_product_search_dense_retrieval.encoders.bi_bert_encoder import BiBERTEncoder
from amazon_product_search_dense_retrieval.encoders.losses import TripletLoss


def test_compute_score():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    bert_model_name = "cl-tohoku/bert-base-japanese-v2"
    bi_encoder = BiBERTEncoder(
        bert_model_name=bert_model_name,
        bert_model_trainable=False,
        rep_mode="cls",
        num_query_projection=4,
        criteria=TripletLoss(),
    )

    score = bi_encoder.compute_score(vec, vec)
    assert score.eq(torch.tensor([1.0, 1.0])).all()

    score = bi_encoder.compute_score(vec, -vec)
    assert score.eq(torch.tensor([-1.0, -1.0])).all()
