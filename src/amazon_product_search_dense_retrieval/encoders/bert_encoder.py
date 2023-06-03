from typing import Literal

import numpy as np
import torch
from more_itertools import chunked
from torch import Tensor
from torch.nn import Linear, Module
from transformers import AutoModel, AutoTokenizer

RepMode = Literal["cls", "mean", "max"]
Target = Literal["query", "doc"]


class BERTEncoder(Module):
    def __init__(
        self,
        bert_model_name: str,
        bert_model_trainable: bool = False,
        rep_mode: RepMode = "mean",
        num_hidden: int = 768,
        num_proj: int = 768,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = bert_model_trainable
        self.rep_mode = rep_mode
        self.query_projection = Linear(num_hidden, num_proj)
        self.doc_projection = Linear(num_hidden, num_proj)

    @staticmethod
    def from_state(
        bert_model_name: str,
        model_filepath: str,
        rep_mode: RepMode = "mean",
        num_hidden: int = 768,
        num_proj: int = 768,
    ) -> "BERTEncoder":
        encoder = BERTEncoder(
            bert_model_name, rep_mode=rep_mode, num_hidden=num_hidden, num_proj=num_proj
        )
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

    @staticmethod
    def convert_token_embs_to_text_emb(
        token_embs: Tensor, attention_mask: Tensor, rep_mode: RepMode
    ) -> Tensor:
        match rep_mode:
            case "cls":
                text_emb = token_embs[:, 0]
            case "mean":
                attention_mask = attention_mask.unsqueeze(dim=-1)
                masked_embeddings = token_embs * attention_mask
                summed = masked_embeddings.sum(dim=1)
                num_non_zero = attention_mask.sum(1).clamp(min=1e-9)
                text_emb = summed / num_non_zero
            case "max":
                text_emb, _ = (token_embs * attention_mask.unsqueeze(dim=-1)).max(dim=1)
            case _:
                raise ValueError(f"Unexpected rep_mode is given: {rep_mode}")
        return text_emb

    def forward(self, tokens: dict[str, Tensor], target: Target) -> Tensor:
        token_embs = self.bert_model(**tokens).last_hidden_state
        text_emb = self.convert_token_embs_to_text_emb(
            token_embs, tokens["attention_mask"], self.rep_mode
        )
        match target:
            case "query":
                text_emb = self.query_projection(text_emb)
            case "doc":
                text_emb = self.doc_projection(text_emb)
            case _:
                raise ValueError()
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
