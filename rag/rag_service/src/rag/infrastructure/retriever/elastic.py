from elasticsearch import AsyncElasticsearch

from rag.domain.document import Dcument
from rag.infrastructure.retriever.base import Retriever


class AsyncElasticsearchRetriever(Retriever):
    def __init__(self, client: AsyncElasticsearch, index: str) -> None:
        self._client = client
        self._index = index

    async def retrieve(self, query: str, top_k: int = 5) -> list[Dcument]:
        resp = await self._client.search(
            index=self._index,
            body={
                "size": top_k,
            },
            # write your impl of ensambling
        )
        return [
            {
                "content": hit["_source"]["content"],
                "metadata": hit["_source"].get("metadata", {}),
            }
            for hit in resp["hits"]["hits"]
        ]
