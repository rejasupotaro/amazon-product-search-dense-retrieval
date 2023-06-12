import torch

from amazon_product_search_dense_retrieval.losses import (
    CombinedLoss,
    InfoNCELoss,
    TripletLoss,
)


def test_triplet_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    criteria = TripletLoss()

    query = vec
    pos_doc = vec
    neg_doc = -vec
    loss = criteria.forward(query, pos_doc, neg_doc)
    assert loss.eq(torch.tensor([0.0, 0.0])).all()

    query = vec
    pos_doc = -vec
    neg_doc = vec
    loss = criteria.forward(query, pos_doc, neg_doc)
    assert loss.eq(torch.tensor([2.4, 2.4])).all()


def test_info_nce_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    criteria = InfoNCELoss()

    query = vec
    pos_doc = vec
    neg_doc = -vec
    low_loss = criteria.forward(query, pos_doc, neg_doc).mean()

    query = vec
    pos_doc = -vec
    neg_doc = vec
    high_loss = criteria.forward(query, pos_doc, neg_doc).mean()
    assert low_loss < high_loss


def test_combined_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    criteria = CombinedLoss(TripletLoss(), InfoNCELoss())

    query = vec
    pos_doc = vec
    neg_doc = -vec
    low_loss = criteria.forward(query, pos_doc, neg_doc).mean()

    query = vec
    pos_doc = -vec
    neg_doc = vec
    high_loss = criteria.forward(query, pos_doc, neg_doc).mean()
    assert low_loss < high_loss
