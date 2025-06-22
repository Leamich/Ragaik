from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains import LLMChain


class Generator(ABC):
    """
    Abstract class for generating answers from contexts.
    """

    @abstractmethod
    def generate(self, query: str, contexts: List[Document] | None) -> str:
        """Generate a response given a query and list of chunk contexts."""
        pass


class RussianPhi4Generator(Generator):
    def __init__(
        self,
        system_prompt: str = "Ты помощник по математике. Отвечай на русском. Решай задачу пошагово (Chain-of-Thoughts). Контест может содержать несколько источников, которые могут быть полезны для ответа на вопрос. Если контекст содержит информацию, которая может помочь в ответе на вопрос, используй её. Если конктест остутствует, то отвечай на основании своих знаний"
    ):
        system_prompt = system_prompt
        llm = OllamaLLM(model="phi4")
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                SystemMessage(content="Контекст:\n{context}")
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
            history_messages_key="history"
        )

    def _get_message_history(self, session_id: str):
        return RedisChatMessageHistory(
            session_id=session_id,
            url="redis://localhost:6379",
            ttl=3600 * 24
        )

    def _format_contexts(self, contexts: List[Document] | None) -> str:
        if not contexts:
            return "Контекст отсутствует."

        return "\n\n".join(
            f"Источник {i + 1}:\n{doc.page_content}" for i, doc in enumerate(contexts)
        )

    def generate(self, query: str, contexts: List[Document] | None) -> str:
        context_block = self._format_contexts(contexts)
        response = self._chain.invoke(
            {"context": context_block, "question": query},
            config={"configurable": {"session_id": "66687"}}
        )
        return str(response)
    


