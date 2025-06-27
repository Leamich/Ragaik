from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi.params import Depends

import RAG.config as config

from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..domain.context_service import ContextService
from ..domain.model_chat_service import ModelChatService
from ..domain.port.loader import DocumentLoader
from ..infrastructure.chunk_repository.bm25_chunk_repository import BM25ChunkRepository
from ..infrastructure.chunk_repository.faiss_chunk_repository import (
    FaissChunkRepository,
)
from ..infrastructure.ollama_llm_chat_adapter import OllamaLLMChatAdapter


@lru_cache()
def get_faiss_chunk_repo(
) -> FaissChunkRepository:
    filename = Path(config.FAISS_CACHE_DIR)
    if not filename.exists():
        raise FileNotFoundError(f"Faiss cache file not found at {filename}")
    return FaissChunkRepository(filename=Path(config.FAISS_CACHE_DIR))

@lru_cache()
def get_bm25_chunk_repo(
) -> BM25ChunkRepository:
    return BM25ChunkRepository(filename=Path(config.BM25_CACHE_FILE))

def get_faiss_and_bm25_ensemble_retriever(
    faiss_chunk_repo: Annotated[FaissChunkRepository, Depends(get_faiss_chunk_repo)],
    bm25_chunk_repo: Annotated[BM25ChunkRepository, Depends(get_bm25_chunk_repo)],
) -> FaissAndBM25EnsembleRetriever:
    return FaissAndBM25EnsembleRetriever(
        faiss_repo=faiss_chunk_repo,
        bm_repo=bm25_chunk_repo,
    )


def get_photos_loader() -> DocumentLoader:
    return None  # Replace with actual photo loader implementation

def get_document_loader() -> DocumentLoader:
    return None  # Replace with actual document loader implementation


@lru_cache
def get_rus_phi4_generator():
    return OllamaLLMChatAdapter(
        model=config.OLLAMA_MODEL_NAME,
        api_url=config.OLLAMA_API_URL,
    )


def get_context_service(
    document_loader: Annotated[DocumentLoader, Depends(get_document_loader)],
    photos_loader: Annotated[DocumentLoader, Depends(get_photos_loader)],
    retriever: Annotated[
        FaissAndBM25EnsembleRetriever, Depends(get_faiss_and_bm25_ensemble_retriever)
    ],
):
    return ContextService(document_loader, photos_loader, retriever)


def get_rag_service(
    context_service: Annotated[ContextService, Depends(get_context_service)],
    rus_phi4_generator: Annotated[
        OllamaLLMChatAdapter, Depends(get_rus_phi4_generator)
    ],
):
    return ModelChatService(
        context_service=context_service,
        generator=rus_phi4_generator,
    )
