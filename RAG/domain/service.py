from typing import Any, List
from .port import DocumentLoader, Chunk, Chunker, DataStore, Retriever, Generator

class RAGService:
    """
    Service for managing RAG documents.
    """
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        datastore: DataStore,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 5,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.datastore = datastore
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def ingest(self, source: Any) -> None:
        """Load documents, chunk them, and add to the datastore."""
        docs = self.loader.load(source)
        chunks: List[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunker.chunk(doc))
        self.datastore.add(chunks)

    def ask(self, query: str) -> str:
        """Retrieve top_k chunks and generate a response."""
        contexts = self.retriever.retrieve(query, self.top_k)
        return self.generator.generate(query, contexts)
