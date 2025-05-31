from typing import Annotated

from fastapi.params import Depends
from ..domain.port import DocumentLoader
from ..domain.service import RAGService
from functools import lru_cache
from ..domain.port.generator import RussianPhi4Generator
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever

def get_cocument_loader() -> DocumentLoader:
    return None  # Replace with actual document loader implementation

@lru_cache
def get_rus_phi4_generator():
    return RussianPhi4Generator()

def faiss_and_bm25_ensemble_retriever():
    return FaissAndBM25EnsembleRetriever()

def get_rag_service(
    document_loader: Annotated[DocumentLoader, Depends(get_cocument_loader)],
    rus_phi4_generator: Annotated[RussianPhi4Generator, Depends(get_rus_phi4_generator)]
):
    return RAGService(
        loader=document_loader,
        generator=rus_phi4_generator,
        chunk_repository=faiss_and_bm25_ensemble_retriever()
    )
