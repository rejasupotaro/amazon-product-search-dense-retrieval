import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from amazon_product_search_dense_retrieval.encoders import (
    BiEncoder,
    ProductEncoder,
    QueryEncoder,
)
from amazon_product_search_dense_retrieval.encoders.modules.pooler import PoolingMode
from amazon_product_search_dense_retrieval.encoders.text_encoder import TextEncoder


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool,
        pooling_mode: PoolingMode,
        criteria: Module,
        lr: float = 1e-4,
    ):
        super().__init__()
        text_encoder = TextEncoder(
            bert_model_name,
            bert_model_trainable,
            pooling_mode,
        )
        query_encoder = QueryEncoder(text_encoder)
        product_encoder = ProductEncoder(text_encoder)
        self.bi_bert_encoder = BiEncoder(
            query_encoder=query_encoder,
            product_encoder=product_encoder,
            criteria=criteria,
        )
        self.lr = lr

    def forward(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        query, pos_doc, neg_doc = batch
        return self.bi_bert_encoder(query, pos_doc, neg_doc)

    def training_step(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]
    ) -> Tensor:
        loss, acc = self(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]],
        batch_idx,
    ) -> Tensor:
        loss, acc = self(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(
        self,
        batch: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]],
        batch_idx,
    ) -> Tensor:
        loss, acc = self(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer
