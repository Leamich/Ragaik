from abc import ABC, abstractmethod

from langchain.schema import Document


class LLMChatAdapter(ABC):
    """
    Abstract class for generating answers from contexts.
    """

    @abstractmethod
    def generate(
        self, query: str, contexts: list[Document], session_id: str | None = None
    ) -> str:
        """Generate a response given a query and list of chunk contexts."""
        pass

    @abstractmethod
    def get_message_history_messages(self, session_id: str) -> list[str]:
        """Get message history for a given session."""
        pass

    @abstractmethod
    def clear_message_history(self, session_id: str) -> None:
        """Clear message history for a given session."""
        pass


