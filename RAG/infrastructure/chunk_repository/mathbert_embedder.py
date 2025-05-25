from typing import List


from transformers import BertTokenizer, BertModel
from langchain_core.embeddings import Embeddings
import torch


class MathBERTEmbedder(Embeddings):
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)
        self._model = BertModel.from_pretrained('tbs17/MathBERT')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self._tokenizer.model_max_length,
        )
        with torch.no_grad():
            outputs = self._model(**tokens)
        vec = outputs.pooler_output[0]
        return vec.cpu().numpy().tolist()
