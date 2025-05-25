from typing import Any

from .port import DocumentLoader, Generator
from chunk_repo_ensemble import EnsembleRetriever


class RAGService:
    """
    Service for managing RAG documents.
    """
    def __init__(
        self,
        loader: DocumentLoader,
        generator: Generator,
        chunk_repository=EnsembleRetriever
    ) -> None:
        self._loader = loader
        self._chunk_repository = chunk_repository
        self._generator = generator

    def ingest(self, source: Any) -> None:
        """Load documents, chunk them, and add to the datastore."""
        documents = self._loader.load(source)
        self._chunk_repository.add_batch(documents)

    def ask(self, query: str) -> str:
        """Retrieve top_k chunks and generate a response."""
        context = self._chunk_repository.query(query)
        return self._generator.generate(query, context)
