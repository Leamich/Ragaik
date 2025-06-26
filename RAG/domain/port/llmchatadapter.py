from abc import ABC, abstractmethod

from langchain.chains import LLMChain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

from RAG.domain.context_service import Context


class LLMChatAdapter(ABC):
    """
    Abstract class for generating answers from contexts.
    """

    @abstractmethod
    def generate(self, query: str, session_id: str, contexts: Context) -> str:
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


class RussianPhi4LLMChatAdapter(LLMChatAdapter):
    def __init__(
        self,
        model: str = "phi4",
        system_prompt: str = "Ты — эксперт по математике. Дай развёрнутый и строго обоснованный ответ на вопрос. Используй определения, теоремы, леммы и примеры. Все формулы и выражения — в формате LaTeX. Контекст (если он есть) имеет приоритет перед общими знаниями. Структура ответа: формулировка задачи, определение необходимых понятий, пошаговое рассуждение, вывод.",
    ):
        llm = OllamaLLM(model=model)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                SystemMessage(content="Контекст:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
        )
        self._chain = RunnableWithMessageHistory(
            chain,
            get_session_history=self._get_message_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    @staticmethod
    def _get_message_history(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=session_id, url="redis://localhost:6379", ttl=3600 * 24
        )

    def get_message_history_messages(self, session_id: str) -> list[str]:
        history = self._get_message_history(session_id)
        return [msg.content for msg in history.messages if isinstance(msg.content, str)]

    def clear_message_history(self, session_id: str) -> None:
        history = self._get_message_history(session_id)
        history.clear()

    @staticmethod
    def _format_contexts(contexts: Context) -> str:
        if not contexts:
            return "Контекст отсутствует."

        return "\n\n".join(
            f"Источник {i + 1}:\n{doc.page_content}" for i, doc in enumerate(contexts)
        )

    def generate(self, query: str, session_id: str, contexts: Context) -> str:
        context_block = self._format_contexts(contexts)
        response = self._chain.invoke(
            {"context": context_block, "question": query},
            config=RunnableConfig(configurable={"session_id": session_id}),
        )
        return response["text"]
