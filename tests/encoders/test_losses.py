import torch

from amazon_product_search_dense_retrieval.encoders.losses import ContrastiveLoss


def test_contrastive_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    query = vec
    pos_doc = vec
    neg_doc = -vec

    criteria = ContrastiveLoss()
    loss = criteria.forward(query, pos_doc, neg_doc)
    assert loss.eq(torch.tensor([0.0, 0.0])).all()

    query = vec
    pos_doc = -vec
    neg_doc = vec

    criteria = ContrastiveLoss()
    loss = criteria.forward(query, pos_doc, neg_doc)
    assert loss.eq(torch.tensor([2.4, 2.4])).all()
