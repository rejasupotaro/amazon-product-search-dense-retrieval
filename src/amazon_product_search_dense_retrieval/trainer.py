import pytorch_lightning as pl
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from amazon_product_search_dense_retrieval.encoders import BiBERTEncoder
from amazon_product_search_dense_retrieval.encoders.bert_encoder import RepMode


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool,
        rep_mode: RepMode,
        projection_shape: tuple[int, int],
        criteria: Module,
    ):
        super().__init__()
        self.bi_bert_encoder = BiBERTEncoder(
            bert_model_name, bert_model_trainable, rep_mode, projection_shape, criteria
        )

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
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer
