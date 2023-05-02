from sentence_transformers import SentenceTransformer
from torch import Tensor


class SBERTEncoder:
    def __init__(self, bert_model_name: str):
        self.sentence_transformer = SentenceTransformer(bert_model_name)

    def encode(self, texts: list[str]) -> Tensor:
        return self.sentence_transformer.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
