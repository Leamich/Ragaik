from .context_service import Context, ContextService
from .port.llmchatadapter import LLMChatAdapter


class ModelChatService:
    """
    Service for managing RAG documents.
    """

    def __init__(
        self,
        context_service: ContextService,
        generator: LLMChatAdapter,
    ) -> None:
        self._generator = generator
        self._context_service = context_service

    def ask(self, query: str, session_id: str) -> tuple[str, list[str]]:
        """Retrieve top_k chunks and generate a response."""
        notes_context: Context = self._context_service.retrieve_textual_context(query)
        photos_context: Context = self._context_service.retrieve_photo_context(query)
        photo_ids = self._context_service.get_context_photo_ids(photos_context)

        return self._generator.generate(query, notes_context, session_id), photo_ids

    def get_history(self, session_id: str) -> list[str]:
        """Get message history for a given session."""
        return self._generator.get_message_history_messages(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear message history for a given session."""
        self._generator.clear_message_history(session_id)
