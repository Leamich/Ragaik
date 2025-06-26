from .context_service import ContextService, Context
from .port import LLMChatAdapter
from .port.llmchatadapter import RussianPhi4LLMChatAdapter

class ModelChatService:
    """
    Service for managing RAG documents.
    """

    def __init__(
        self,
        context_service: ContextService,
        generator: LLMChatAdapter = RussianPhi4LLMChatAdapter(),
    ) -> None:
        self._generator = generator
        self._context_service = context_service

    def ask(self, query: str, session_id: str) -> tuple[str, str | None]:
        """Retrieve top_k chunks and generate a response."""
        notes_context: Context = self._context_service.retrieve_photo_context(query)
        photos_context: Context = self._context_service.retrieve_photo_context(query)
        photo_ids = self._context_service.get_context_photo_ids(photos_context)
        photo_id = photo_ids[0] if photo_ids else None

        return self._generator.generate(query, session_id, notes_context), photo_id

    def get_history(self, session_id: str) -> list[str]:
        """Get message history for a given session."""
        return self._generator.get_message_history_messages(session_id)

    def clear_history(self, session_id: str) -> None:
        """Clear message history for a given session."""
        self._generator.clear_message_history(session_id)
