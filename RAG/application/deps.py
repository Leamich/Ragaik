from functools import lru_cache
from typing import Annotated

from fastapi.params import Depends

import RAG.config as config

from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..domain.context_service import ContextService
from ..domain.model_chat_service import ModelChatService
from ..domain.port.loader import DocumentLoader
from ..infrastructure.ollama_llm_chat_adapter import OllamaLLMChatAdapter


def get_cocument_loader() -> DocumentLoader:
    return None  # Replace with actual document loader implementation


def get_photos_loader() -> DocumentLoader:
    return None  # Replace with actual photo loader implementation


@lru_cache
def get_rus_phi4_generator():
    return OllamaLLMChatAdapter(
        model=config.OLLAMA_MODEL_NAME,
        api_url=config.OLLAMA_API_URL,
    )


def faiss_and_bm25_ensemble_retriever():
    return FaissAndBM25EnsembleRetriever()


def get_context_service(
    document_loader: Annotated[DocumentLoader, Depends(get_cocument_loader)],
    photos_loader: Annotated[DocumentLoader, Depends(get_photos_loader)],
    retriever: Annotated[
        FaissAndBM25EnsembleRetriever, Depends(faiss_and_bm25_ensemble_retriever)
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
