from .port.llmchatadapter import LLMChatAdapter
from ..retrivers import EnsembleRetriever



class ModelChatService:
    """
    Service for managing RAG documents.
    """

    def __init__(
        self,
        generator: LLMChatAdapter,
        EnsembleRetriever: EnsembleRetriever
    ) -> None:
        self._generator = generator
        self._retriever = EnsembleRetriever

    async def ask(self, query: str, session_id: str) -> tuple[str, list[str]]:
        """Retrieve top_k chunks and generate a response."""
        context = await self._retriever.query(query)
        photo_ids = [doc.metadata["image_id"] for doc in context]

        return self._generator.generate(query, context, session_id), photo_ids

    def get_history(self, session_id: str) -> list[str]:
        """Get message history for a given session."""
        return self._generator.get_message_history_messages(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear message history for a given session."""
        self._generator.clear_message_history(session_id)
