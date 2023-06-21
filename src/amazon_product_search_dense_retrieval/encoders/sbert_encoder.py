
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from torch import Tensor

from amazon_product_search_dense_retrieval.encoders.pooler import PoolingMode


class SBERTEncoder:
    def __init__(
        self, bert_model_name: str, pooling_mode: PoolingMode = "mean"
    ) -> None:
        transformer = Transformer(bert_model_name)
        pooling = Pooling(
            word_embedding_dimension=transformer.get_word_embedding_dimension(),
            pooling_mode=pooling_mode,
        )
        self.sentence_transformer = SentenceTransformer(modules=[transformer, pooling])

    def encode(self, texts: list[str]) -> Tensor:
        return self.sentence_transformer.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
