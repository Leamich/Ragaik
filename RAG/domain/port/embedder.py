from abc import ABC, abstractmethod
from typing import List

from transformers import BertTokenizer, TFBertModel
from torch import FloatTensor

from ..chunk import Chunk


class Embedder(ABC):
    """
    Abstract port class for generating embeddings from texts.
    Supposed to embed one Document, split into chunks.
    """
    @abstractmethod
    def embed(self, chunk: Chunk) -> tuple[FloatTensor]:
        """Embed a list of texts and return corresponding embeddings."""
        pass


class MathBERTEmbedder(Embedder):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)
        self._model = TFBertModel.from_pretrained('tbs17/MathBERT')

    def embed(self, chunk: Chunk) -> tuple[FloatTensor]:
        tokens = self._tokenizer(chunk, return_tensors='tf')
        return self._model(tokens)

