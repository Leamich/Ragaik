from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain_community.llms import Ollama

class Generator(ABC):
    """
    Abstract port class for generating answers from contexts.
    """
    @abstractmethod
    def generate(self, query: str, contexts: List[Document]) -> str:
        """Generate a response given a query and list of chunk contexts."""
        pass

class RussianPhi4Generator(Generator):
    def __init__(self, system_prompt: str = "Ты помощник по математике. Отвечай на русском и, по возможности, ссылайся на представленный контекст. Решай задачу пошагово (Chain-of-Thoughts)."):
        self._system_prompt = system_prompt
        self._llm = Ollama(model="phi4")

    def _format_prompt(self, query: str, contexts: List[Document] | None) -> str:
        if contexts is None:
            context_texts = "Контекст отсутствует."
        
        else:
            context_texts = "\n\n".join([
                f"Источник {i+1} ({doc.metadata.get('url', 'неизвестно')}):\n{doc.page_content}"
                for i, doc in enumerate(contexts)
            ])

        return (
            f"{self._system_prompt}\n\n"
            f"Контекст:\n{context_texts}\n\n"
            f"Вопрос: {query}\n"
            f"Ответ:"
        )

    def generate(self, query: str, contexts: List[Document] | None) -> str:
        prompt = self._format_prompt(query, contexts)
        raw_response = self._llm.invoke(prompt)

        if contexts is None:
            return raw_response
        
        urls = [doc.metadata.get("url") for doc in contexts if "url" in doc.metadata]
        unique_urls = list(set(urls))
        sources_text = "\n\nИсточники:\n" + "\n".join(unique_urls) if unique_urls else ""
        return raw_response + sources_text
