import torch

from amazon_product_search_dense_retrieval.encoders.bi_encoder import (
    BiEncoder,
    ProductEncoder,
    QueryEncoder,
)
from amazon_product_search_dense_retrieval.encoders.modules.losses import TripletLoss
from amazon_product_search_dense_retrieval.encoders.text_encoder import TextEncoder


def test_compute_score():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    text_encoder = TextEncoder(
        hf_model_name="cl-tohoku/bert-base-japanese-v2",
        hf_model_trainable=False,
        pooling_mode="cls",
    )
    query_encoder = QueryEncoder(text_encoder)
    product_encoder = ProductEncoder(text_encoder)
    bi_encoder = BiEncoder(
        query_encoder=query_encoder,
        product_encoder=product_encoder,
        criteria=TripletLoss(),
    )

    score = bi_encoder.compute_score(vec, vec)
    assert score.eq(torch.tensor([1.0, 1.0])).all()

    score = bi_encoder.compute_score(vec, -vec)
    assert score.eq(torch.tensor([-1.0, -1.0])).all()
