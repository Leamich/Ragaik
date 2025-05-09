from transformers import BertTokenizer, TFBertModel

from domain.chunk import Chunk
from infrastructure.chunk_repository.embedder import AbstractEmbedder, Embedding


class MathBERTEmbedder(AbstractEmbedder):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)
        self._model = TFBertModel.from_pretrained('tbs17/MathBERT')

    def embed(self, chunk: Chunk) -> Embedding:
        tokens = self._tokenizer(chunk, return_tensors='tf')
        return self._model(tokens)
