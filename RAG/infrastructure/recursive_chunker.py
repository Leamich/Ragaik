from langchain.text_splitter import HuggingFaceTokenTextSplitter
from langchain.schema import Document
from .chunk_repository.chunker import Chunker


class TokenChunker(Chunker):
    """
    Recursive realization of Chunker using tokenizer compatible with multilingual-e5-large.
    Chunk size = 400 tokens by default.
    """

    def __init__(
        self,
        tokenizer_name: str = "intfloat/multilingual-e5-large",
        chunk_size: int = 400
    ) -> None:
        self._CHUNK_SIZE = chunk_size
        self._tokenizer_name = tokenizer_name

        self._text_splitter = HuggingFaceTokenTextSplitter(
            encoding_name=self._tokenizer_name,
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
            model_name=self._tokenizer_name
        )

    def chunk(self, document: Document) -> list[Document]:
        return self._text_splitter.split_documents([document])

