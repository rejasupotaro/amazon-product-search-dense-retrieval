from typing import Literal

import numpy as np
import torch
from more_itertools import chunked
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from amazon_product_search_dense_retrieval.encoders.modules.pooler import (
    Pooler,
    PoolingMode,
)

Target = Literal["query", "doc"]
ProjectionMode = Literal["none", "query", "doc", "both"]


class BERTEncoder(Module):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool,
        pooling_mode: PoolingMode,
        projection_mode: ProjectionMode,
        projection_shape: tuple[int, int],
    ) -> None:
        super().__init__()
        self.bert_model_name = bert_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = bert_model_trainable
        self.pooler = Pooler(pooling_mode)
        self.projection_mode = projection_mode
        self.projection_shape = projection_shape
        self.projection = Linear(*projection_shape)

    @staticmethod
    def build_model_name(
        bert_model_name: str,
        pooling_mode: PoolingMode,
        projection_mode: ProjectionMode,
        projection_shape: tuple[int, int],
    ):
        bert_model_name = bert_model_name.replace("/", "_")
        return f"{bert_model_name}_{pooling_mode}_{projection_mode}_{projection_shape[0]}_{projection_shape[1]}"

    def save(self, models_dir: str) -> str:
        model_name = self.build_model_name(
            self.bert_model_name,
            self.pooler.pooling_mode,
            self.projection_mode,
            self.projection_shape,
        )
        model_filepath = f"{models_dir}/{model_name}.pt"
        torch.save(self.projection.state_dict(), model_filepath)
        return model_filepath

    @staticmethod
    def load(
        bert_model_name: str,
        bert_model_trainable: bool,
        pooling_mode: PoolingMode,
        projection_mode: ProjectionMode,
        projection_shape: tuple[int, int],
        models_dir: str,
    ) -> "BERTEncoder":
        encoder = BERTEncoder(
            bert_model_name,
            bert_model_trainable,
            pooling_mode,
            projection_mode,
            projection_shape,
        )
        model_name = BERTEncoder.build_model_name(
            bert_model_name, pooling_mode, projection_mode, projection_shape
        )
        model_filepath = f"{models_dir}/{model_name}.pt"
        encoder.projection.load_state_dict(torch.load(model_filepath))
        return encoder

    def tokenize(self, texts) -> dict[str, Tensor]:
        tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokens

    def forward(self, tokens: dict[str, Tensor], target: Target) -> Tensor:
        token_embs = self.bert_model(**tokens).last_hidden_state
        text_emb = self.pooler.forward(token_embs, tokens["attention_mask"])
        if (target == "query" and self.projection_mode in ["query", "both"]) or (
            target == "doc" and self.projection_mode in ["doc", "both"]
        ):
            text_emb = self.projection(text_emb)
        text_emb = normalize(text_emb, p=2, dim=1)
        return text_emb

    def encode(
        self, texts: str | list[str], target: Target, batch_size: int = 32
    ) -> np.ndarray:
        input_was_string = False
        if isinstance(texts, str):
            input_was_string = True
            texts = texts

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.eval()
        all_embs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in chunked(texts, n=batch_size):
                tokens = self.tokenize(batch)
                for key in tokens:
                    if isinstance(tokens[key], Tensor):
                        tokens[key] = tokens[key].to(device)
                embs: Tensor = self.forward(tokens, target)
                embs = embs.detach().cpu().numpy()
                all_embs.extend(embs)
        return np.array(all_embs)[0] if input_was_string else np.array(all_embs)

    def encode_query(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, "query", batch_size)

    def encode_doc(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, "doc", batch_size)
