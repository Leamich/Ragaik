""" "File with Elasticsearch DocumentSore implementation."""

from langchain_elasticsearch import AsyncElasticsearchStore
from langchain.retrievers import EnsembleRetriever
from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .base import DocumentStore


class AsyncElasticsearchRetriever(DocumentStore):
    """Elasticsearch DocumentSore implementation."""

    def __init__(
        self,
        embeddings: Embeddings,
        connection: AsyncElasticsearch,
        index_name: str,
        hybrid_alpha: float = 0.7,
    ) -> None:
        """Initialize Elasticsearch retriever with hybrid search.

        Args:
            embeddings: Embeddings model
            client: AsyncElasticsearch client
            index: Name of the Elasticsearch index
            hybrid_alpha: Weight for vector similarity vs BM25 (0.0 to 1.0)
                         0.0 = pure BM25, 1.0 = pure vector search
        """
        self._store = AsyncElasticsearchStore(
            es_connection=connection,
            index_name=index_name,
            embedding=embeddings,
        )

        bm25_retriever = self._store.as_retriever(
            retrieval_strategy=AsyncElasticsearchStore.BM25RetrievalStrategy()
        )
        vector_retriever = self._store.as_retriever(
            retrieval_strategy=AsyncElasticsearchStore.ApproxRetrievalStrategy()
        )

        self._ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[hybrid_alpha, 1 - hybrid_alpha],
        )

    async def ainvoke(self, query: str, top_k) -> list[Document]:
        return await self._ensemble_retriever.ainvoke(query, top_k)

    async def aadd_documents(self, documents: list[Document]) -> None:
        await self._store.aadd_documents(documents)
