from typing import Protocol

from rag.domain.document import Dcument


class Retriever(Protocol):
    """Abstract retriever interface"""

    async def retrieve(self, query: str, top_k: int = 5) -> list[Dcument]: ...
