from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizer
from transformers import BertTokenizer


from langchain.schema import Document
from .chunk_repository.chunker import Chunker


class RecursiveChunker(Chunker):
    """
    Recursive realization of chunker via given tokenizer.
    Chunk size = 512 tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer
    = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)) -> None:
        self._CHUNK_SIZE: int = 450
        self._tokenizer: PreTrainedTokenizer = tokenizer
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

    def chunk(self, document: Document) -> list[Document]:
        return self._text_splitter.split_documents([document])
