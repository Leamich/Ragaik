from torch import FloatTensor
from transformers import BertTokenizer, TFBertModel

from domain.chunk import Chunk
from domain.port.embedder import Embedder


class MathBERTEmbedder(Embedder):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)
        self._model = TFBertModel.from_pretrained('tbs17/MathBERT')

    def embed(self, chunk: Chunk) -> tuple[FloatTensor]:
        tokens = self._tokenizer(chunk, return_tensors='tf')
        return self._model(tokens)
