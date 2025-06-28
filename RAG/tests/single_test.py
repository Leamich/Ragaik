from ragas import SingleTurnSample
from ragas.metrics import (
    AnswerRelevancy,
    ContextRelevance,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
    Faithfulness,
)
from langchain_ollama import OllamaLLM
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

import pandas as pd


async def main():
    answers = pd.read_pickle("FaissAndBM25EnsembleRetriever_Phi35_res.pkl")
    ff = answers.iloc[1]

    critic_llm = OllamaLLM(model="phi4")
    wrapped_critic = LangchainLLMWrapper(critic_llm)

    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embedder_wrapped = LangchainEmbeddingsWrapper(embedder)

    sample = SingleTurnSample(
        user_input=ff["user_input"],
        response=ff["response"],
        retrieved_contexts=ff["retrieved_contexts"],
    )

    print(
        await AnswerRelevancy(
            llm=wrapped_critic, embeddings=embedder_wrapped
        ).single_turn_ascore(sample),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
