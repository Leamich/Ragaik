from domain.chunk import Document, Chunk
from domain.port import ChunkRepository


class ChunkRepositoryEnsemble:
    """
    A class to manage a collection of chunk repositories.
    """

    def __init__(self, repository_a: ChunkRepository, repository_b: ChunkRepository) -> None:
        """
        Initialize the ensemble with two chunk repositories.
        """
        self._repository_a = repository_a
        self._repository_b = repository_b

    def add(self, document: Document) -> None:
        """
        Add a document to both repositories.
        """
        self._repository_a.add(document)
        self._repository_b.add(document)

    def add_batch(self, documents: list[Document]) -> None:
        """
        Add a batch of documents to both repositories.
        """
        [self.add(document) for document in documents]

    def query(self, key: any, top_k: int) -> tuple[list[Chunk], list[Chunk]]:
        """
        Query both repositories and return the results.
        """
        result_a = self._repository_a.query(key, top_k)
        result_b = self._repository_b.query(key, top_k)
        return result_a, result_b
