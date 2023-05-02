from typing import Literal, Optional

import numpy as np
import torch
from more_itertools import chunked
from torch import Tensor
from torch.nn import Linear, Module, Sequential
from transformers import AutoModel, AutoTokenizer


class BERTEncoder(Module):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool = False,
        rep_mode: Literal["mean", "max", "cls"] = "mean",
        num_hidden: int = 768,
        num_proj: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = bert_model_trainable
        self.rep_mode = rep_mode
        self.projection: Optional[Sequential] = None
        if num_proj:
            self.projection = Sequential(
                Linear(num_hidden, num_proj),
            )

    @staticmethod
    def from_state(bert_model_name: str, model_filepath) -> "BERTEncoder":
        encoder = BERTEncoder(bert_model_name)
        encoder.load_state_dict(torch.load(model_filepath))
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

    def convert_token_embs_to_text_emb(self, token_embs: Tensor, attention_mask: Tensor) -> Tensor:
        if self.rep_mode == "max":
            text_emb, _ = (token_embs * attention_mask.unsqueeze(-1)).max(dim=1)
        elif self.rep_mode == "mean":
            text_emb = (token_embs * attention_mask.unsqueeze(-1)).mean(dim=1)
        elif self.rep_mode == "cls":
            text_emb = token_embs[:, 0]
        else:
            raise ValueError(f"Unexpected rep_mode is given: {self.rep_mode}")
        return text_emb

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        token_embs = self.bert_model(**tokens).last_hidden_state
        text_emb = self.convert_token_embs_to_text_emb(token_embs, tokens["attention_mask"])
        if self.projection:
            text_emb = self.projection(text_emb)
        return torch.nn.functional.normalize(text_emb, p=2, dim=1)

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        self.eval()
        all_embs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in chunked(texts, n=batch_size):
                tokens = self.tokenize(batch)
                embs: Tensor = self(tokens)
                embs = embs.detach().cpu().numpy()
                all_embs.extend(embs)
        return np.array(all_embs)
