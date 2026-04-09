import hashlib

from langchain_text_splitters.base import TokenTextSplitter
from transformers import AutoTokenizer
from langchain_core.documents import Document
from ._chunker import Chunker

#TODO: change tokenizer
class TokenChunker(Chunker):
    """
    Recursive realization of Chunker using tokenizer compatible with multilingual-e5-large.
    Chunk size = 400 tokens by default.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large"),
        chunk_size: int = 400
    ) -> None:
        self._CHUNK_SIZE = chunk_size
        self._tokenizer = tokenizer


        self._text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self._tokenizer,
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self._text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata['chunk_id'] =  hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        return chunks

