from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer

from domain.chunk import Chunk, Document
from infrastructure.chunk_repository.chunker import Chunker


class RecursiveChunker(Chunker):
    """
    Recursive realization of chunker via given tokenizer.
    Chunk size = 512 tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self._CHUNK_SIZE: int = 512
        self._tokenizer: PreTrainedTokenizer = tokenizer
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        langchain_document = LangchainDocument(page_content=document.content)
        chunk_as_documents: list[LangchainDocument] = self._text_splitter.split_documents([langchain_document])
        return [Chunk(chunk.page_content, chunk.metedata) for chunk in chunk_as_documents]
