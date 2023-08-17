import pytorch_lightning as pl
from torch import Tensor
from torch.optim import AdamW

from amazon_product_search_dense_retrieval.encoders import (
    BiEncoder,
)


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        bi_encoder: BiEncoder,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.bi_encoder = bi_encoder
        self.lr = lr

    def forward(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        query, pos_doc, neg_doc = batch
        return self.bi_encoder(query, pos_doc, neg_doc)

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
