from typing import Annotated

from fastapi.params import Depends
from ..domain.port import DocumentLoader
from ..domain.model_chat_service import ModelChatService
from functools import lru_cache
from ..domain.port.llmchatadapter import RussianPhi4LLMChatAdapter
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever

def get_cocument_loader() -> DocumentLoader:
    return None  # Replace with actual document loader implementation

def get_photos_loader() -> DocumentLoader:
    return None  # Replace with actual photo loader implementation

@lru_cache
def get_rus_phi4_generator():
    return RussianPhi4LLMChatAdapter()

def faiss_and_bm25_ensemble_retriever():
    return FaissAndBM25EnsembleRetriever()

def get_rag_service(
    document_loader: Annotated[DocumentLoader, Depends(get_cocument_loader)],
    photos_loader: Annotated[DocumentLoader, Depends(get_photos_loader)],
    rus_phi4_generator: Annotated[RussianPhi4LLMChatAdapter, Depends(get_rus_phi4_generator)] #TODO add loader for photos
):
    return ModelChatService(
        notes_loader=document_loader,
        photos_loader=photos_loader,  #Todo add actual photo loader
        generator=rus_phi4_generator,
        notes_repository=faiss_and_bm25_ensemble_retriever()
    )
