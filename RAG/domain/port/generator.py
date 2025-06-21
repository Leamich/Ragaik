from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM


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
        system_prompt: str = "Ты помощник по математике. Отвечай на русском. Решай задачу пошагово (Chain-of-Thoughts).",
    ):
        self._system_prompt = system_prompt
        self._llm = OllamaLLM(model="phi4")
        self._prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._system_prompt),
            SystemMessage(content="Контекст:\n{context}"),
            HumanMessage(content="{question}")
        ])

    def _format_contexts(self, contexts: List[Document] | None) -> str:
        if not contexts:
            return "Контекст отсутствует."

        return "\n\n".join(
            f"Источник {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(contexts)
        )

    def generate(self, query: str, contexts: List[Document] | None) -> str:
        context_block = self._format_contexts(contexts)
        messages = self._prompt_template.invoke({
            "context": context_block,
            "question": query
        })

        return self._llm.invoke(messages)
