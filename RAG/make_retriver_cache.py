import glob
import os
import pickle
from pathlib import Path

from langchain_core.documents import Document

import RAG.config as config
from RAG.infrastructure.chunk_repository.faiss_chunk_repository import (
    FaissChunkRepository,
)


def find_md_file_paths(root_path: str) -> list[str]:
    glob_path = os.path.join(root_path, "**", "*.md")
    return glob.glob(glob_path, recursive=True)


def load_doc(doc_path: str) -> Document:
    with open(doc_path, "r") as doc:
        return Document(page_content=doc.read(), metadata={"url": doc_path})


def load_documents(root_path: str) -> list[Document]:
    paths = find_md_file_paths(root_path)
    return [load_doc(path) for path in paths]

def make_faiss_cache(docs: list[Document], cache_path: Path) -> None:
    faiss_doc_store = FaissChunkRepository(documents=docs)
    faiss_doc_store.store(cache_path)

def make_mb25_cache(docs: list[Document], cache_path: Path) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump(docs, f)


if __name__ == "__main__":
    root_path = "RAG/tests/hse_conspects_course1/"
    docs = load_documents(root_path)

    faiss_cache_path = Path(config.FAISS_CACHE_DIR)
    # make_faiss_cache(docs, faiss_cache_path)

    bm25_cache_path = Path(config.BM25_CACHE_FILE)
    make_mb25_cache(docs, bm25_cache_path)

    print(f"FAISS cache created at {faiss_cache_path}")
    print(f"BM25 cache created at {bm25_cache_path}")