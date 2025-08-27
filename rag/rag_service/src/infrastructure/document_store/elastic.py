""" "File with Elasticsearch DocumentSore implementation."""

from langchain_elasticsearch import AsyncElasticsearchStore
from langchain.retrievers import EnsembleRetriever
from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .base import DocumentStore


class ElasticsearchDocumentStore(DocumentStore):
    """Elasticsearch DocumentSore implementation."""

    def __init__(
        self, store: AsyncElasticsearchStore, ensemble_retriever: EnsembleRetriever
    ) -> None:
        """ "Support function for initialization. Use init instead."""
        self._store = store
        self._ensemble_retriever = ensemble_retriever

    @classmethod
    async def init(
        cls,
        embedder: Embeddings,
        connection: AsyncElasticsearch,
        index_name: str,
        hybrid_alpha: float = 0.7,
    ) -> "ElasticsearchDocumentStore":
        """Initialize Elasticsearch retriever with hybrid search.

        Args:
            embeddings: Embeddings model
            client: AsyncElasticsearch client
            index: Name of the Elasticsearch index
            hybrid_alpha: Weight for vector similarity vs BM25 (0.0 to 1.0)
                         0.0 = pure BM25, 1.0 = pure vector search
        """

        dims: int = len(await embedder.aembed_query("Тест"))
        await cls._index_creation(connection, index_name, dims=dims)

        store = AsyncElasticsearchStore(
            es_connection=connection,
            index_name=index_name,
            embedding=embedder,
        )

        bm25_retriever = store.as_retriever(
            retrieval_strategy=AsyncElasticsearchStore.BM25RetrievalStrategy()
        )
        vector_retriever = store.as_retriever(
            retrieval_strategy=AsyncElasticsearchStore.ApproxRetrievalStrategy()
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[hybrid_alpha, 1 - hybrid_alpha],
        )

        return ElasticsearchDocumentStore(store, ensemble_retriever)

    @staticmethod
    async def _index_creation(
        connection: AsyncElasticsearch, index_name: str, dims: int
    ) -> None:
        """Creating Elasticsearch database structure."""
        if not await connection.indices.exists(index=index_name):
            await connection.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "vector": {"type": "dense_vector", "dims": dims},
                        }
                    }
                },
            )

    async def ainvoke(self, query: str, top_k) -> list[Document]:
        return await self._ensemble_retriever.ainvoke(query, top_k)

    async def aadd_documents(self, documents: list[Document]) -> None:
        await self._store.aadd_documents(documents)
