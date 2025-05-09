from typing import Any

from .chunk_repo_ensemble import ChunkRepositoryEnsemble
from .port import DocumentLoader, Generator


class RAGService:
    """
    Service for managing RAG documents.
    """
    def __init__(
        self,
        loader: DocumentLoader,
            chunk_repository_ensemble: ChunkRepositoryEnsemble,
        generator: Generator,
    ) -> None:
        self.loader = loader
        self.chunk_repository_ensemble = chunk_repository_ensemble
        self.generator = generator

    def ingest(self, source: Any) -> None:
        """Load documents, chunk them, and add to the datastore."""
        documents = self.loader.load(source)
        self.chunk_repository_ensemble.add_batch(documents)

    def ask(self, query: str, top_k: int = 5) -> str:
        """Retrieve top_k chunks and generate a response."""
        context_a, context_b = self.chunk_repository_ensemble.query(query, top_k)
        # FIXME how to handle the two contexts?
        return self.generator.generate(query, context_a)
