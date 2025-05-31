from transformers import AutoTokenizer, PreTrainedTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .chunk_repository.chunker import Chunker


class RecursiveChunker(Chunker):
    """
    Recursive realization of Chunker using tokenizer compatible with multilingual-e5-large.
    Chunk size = 450 tokens by default.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large"),
        chunk_size: int = 450
    ) -> None:
        self._CHUNK_SIZE = chunk_size
        self._tokenizer: PreTrainedTokenizer = tokenizer

        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self._tokenizer,
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
            add_start_index=True,
            strip_whitespace=True
        )

    def chunk(self, document: Document) -> list[Document]:
        return self._text_splitter.split_documents([document])
