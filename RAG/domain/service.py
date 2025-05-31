from typing import Any, List

from langchain.schema import Document

from .port import DocumentLoader, Generator
from .chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from .port.generator import RussianPhi4Generator


class RAGService:
    """
    Service for managing RAG documents.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        generator: Generator = RussianPhi4Generator(),
        chunk_repository=FaissAndBM25EnsembleRetriever(),
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
        # TODO add chat context
        context: List[Document] | None = self._chunk_repository.query(query)

        return self._generator.generate(query, context)


if __name__ == "__main__":
    test = RAGService(loader=None)
    print(test.ask("Реши уравнение квадратное уравнение: 17x^2 + 6x > 0."))
