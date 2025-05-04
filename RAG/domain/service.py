from typing import Any, List
from .port import DocumentLoader, Chunk, Chunker, ChunkRepository, Generator

class RAGService:
    """
    Service for managing RAG documents.
    """
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        chunk_repository: ChunkRepository,
        generator: Generator,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.chunk_repository = chunk_repository
        self.generator = generator

    def ingest(self, source: Any) -> None:
        """Load documents, chunk them, and add to the datastore."""
        docs = self.loader.load(source)
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunker.chunk(doc))
        self.chunk_repository.add(chunks)

    def ask(self, query: str, top_k: int = 5) -> str:
        """Retrieve top_k chunks and generate a response."""
        contexts = self.chunk_repository.query(query, top_k)
        return self.generator.generate(query, contexts)
