import pickle
from pathlib import Path

from langchain_core.documents import Document

import RAG.config as config
from RAG.infrastructure.chunk_repository.faiss_chunk_repository import (
    FaissChunkRepository,
)
from .load import (
    load_photo_docs, 
    load_documents
)

def make_faiss_cache(docs: list[Document], cache_path: Path) -> None:
    faiss_doc_store = FaissChunkRepository(documents=docs)
    faiss_doc_store.store(cache_path)

def make_mb25_cache(docs: list[Document], cache_path: Path) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(docs, f)


if __name__ == "__main__":
    root_path = Path(config.NOTES_START_DIR)
    docs = load_documents(root_path)

    faiss_cache_path = Path(config.FAISS_CACHE_DIR)
    make_faiss_cache(docs, faiss_cache_path)

    bm25_cache_path = Path(config.BM25_CACHE_FILE)
    make_mb25_cache(docs, bm25_cache_path)

    bm25_photo_cache_path = Path(config.PHOTO_CONTEXT_CACHE)

    phot_text_content_file  = Path(config.PHOT_TEXT_CONTENT_FILE)
    photo_docs = load_photo_docs(phot_text_content_file)
    make_faiss_cache(photo_docs, bm25_photo_cache_path)