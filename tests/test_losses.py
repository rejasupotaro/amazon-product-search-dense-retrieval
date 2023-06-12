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
    low_loss = criteria.forward(query, pos_doc, neg_doc)
    assert low_loss.shape == (2,)

    query = vec
    pos_doc = -vec
    neg_doc = vec
    high_loss = criteria.forward(query, pos_doc, neg_doc)
    assert high_loss.shape == (2,)
    assert low_loss.mean() < high_loss.mean()


def test_info_nce_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    criteria = InfoNCELoss()

    query = vec
    pos_doc = vec
    neg_doc = -vec
    low_loss = criteria.forward(query, pos_doc, neg_doc)
    assert low_loss.shape == (2,)

    query = vec
    pos_doc = -vec
    neg_doc = vec
    high_loss = criteria.forward(query, pos_doc, neg_doc)
    assert high_loss.shape == (2,)
    assert low_loss.mean() < high_loss.mean()


def test_combined_loss():
    vec = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    criteria = CombinedLoss(TripletLoss(), InfoNCELoss())

    query = vec
    pos_doc = vec
    neg_doc = -vec
    low_loss = criteria.forward(query, pos_doc, neg_doc)
    assert low_loss.shape == (2,)

    query = vec
    pos_doc = -vec
    neg_doc = vec
    high_loss = criteria.forward(query, pos_doc, neg_doc)
    assert high_loss.shape == (2,)
    assert low_loss.mean() < high_loss.mean()
