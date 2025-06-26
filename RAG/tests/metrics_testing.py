import pandas as pd
from ragas.metrics import (
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from ragas.utils import Dataset
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..domain.port.llmchatadapter import RussianPhi4LLMChatAdapter
from .load_local import load_documents

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


def run_ragas_evaluation(
    system_name: str, retriever: FaissAndBM25EnsembleRetriever, generator
):
    critic_llm = OllamaLLM(model="phi3.5")
    wrapped_critic = LangchainLLMWrapper(critic_llm)

    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embedder_wrapped = LangchainEmbeddingsWrapper(embedder)

    run_config = RunConfig(max_workers=1, timeout=500)

    with open("RAG/tests/questions.md") as f:
        questions = [s[3:] for s in f.readlines()]

    datasamples = {"question": [], "answer": [], "contexts": []}

    for question in questions:
        contexts = retriever.query(question)
        if contexts is None:
            contexts = []
        generated = generator.generate(question, contexts)

        datasamples["question"].append(question)
        datasamples["answer"].append(generated)
        datasamples["contexts"].append([doc.page_content for doc in contexts])

    dataset = Dataset.from_dict(datasamples)
    results: pd.DataFrame = evaluate(
        dataset,
        metrics=[
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
        ],
        raise_exceptions=False,
        llm=wrapped_critic,
        embeddings=embedder_wrapped,
        run_config=run_config,
    ).to_pandas()

    results.to_pickle("res.pkl")
    results.to_excel("res.xlsx")


def evaluate_faiss_bm25_phi4():
    retriever = FaissAndBM25EnsembleRetriever()
    generator = RussianPhi4LLMChatAdapter()

    start_path = "RAG/tests/hse_conspects_course1/"
    documents = load_documents(start_path)
    retriever.add_batch(documents)

    run_ragas_evaluation("FaissAndBM25EnsembleRetriever + Phi4", retriever, generator)


if __name__ == "__main__":
    evaluate_faiss_bm25_phi4()
