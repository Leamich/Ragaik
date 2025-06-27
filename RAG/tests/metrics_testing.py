import os
from pathlib import Path
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy
)
from ragas.run_config import RunConfig
from ragas.utils import Dataset
from tqdm import tqdm

from RAG.domain.port.llmchatadapter import LLMChatAdapter
from RAG.infrastructure.ollama_llm_chat_adapter import OllamaLLMChatAdapter

from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..load import load_documents
import RAG.config as config


def run_ragas_evaluation(
    system_name: str, retriever: FaissAndBM25EnsembleRetriever, generator: LLMChatAdapter
):
    print("INFO: initializing critic LLM")
    critic_llm = OllamaLLM(model="phi4")
    wrapped_critic = LangchainLLMWrapper(critic_llm)

    print("INFO: initializing embedder")
    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embedder_wrapped = LangchainEmbeddingsWrapper(embedder)

    run_config = RunConfig(max_workers=1, timeout=500)

    print("INFO: loading questions")
    with open("RAG/tests/questions.md") as f:
        questions = [s[3:] for s in f.readlines()]

    print("INFO: loaded", len(questions), "questions")

    datasamples = {"question": [], "answer": [], "contexts": []}

    if not os.path.exists(f"{system_name}_answers.pkl"):
        for question in tqdm(questions, desc="Generating answers"):
            contexts = retriever.query(question)
            generated = generator.generate(question, contexts)

            datasamples["question"].append(question)
            datasamples["answer"].append(generated)
            datasamples["contexts"].append([doc.page_content for doc in contexts])

        answers = pd.DataFrame.from_dict(datasamples)
        answers.to_pickle(f"{system_name}_answers.pkl")
        answers.to_excel(f"{system_name}_answers.xlsx")
        dataset = Dataset.from_dict(datasamples)

    else:
        answers = pd.read_pickle(f"{system_name}_answers.pkl")
        dataset = Dataset.from_pandas(answers)


    print("INFO: generating dataset")
    results: pd.DataFrame = evaluate(
        dataset,
        metrics=[
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
            Faithfulness()
        ],
        raise_exceptions=False,
        llm=wrapped_critic,
        embeddings=embedder_wrapped,
        run_config=run_config,
    ).to_pandas()

    results.to_pickle(f"{system_name}_res.pkl")
    results.to_excel(f"{system_name}_res.xlsx")


def evaluate_faiss_bm25_phi4():
    print("INFO: initializing retriever")
    retriever = FaissAndBM25EnsembleRetriever()

    print("INFO: initializing generator")
    generator = OllamaLLMChatAdapter()

    print("INFO: loading documents")
    start_path = Path(config.NOTES_START_FILE)
    documents = load_documents(start_path)
    print("INFO: loaded", len(documents), "documents")

    print("INFO: adding documents to retriever")
    retriever.add_batch(documents)

    print("INFO: running evaluation")
    run_ragas_evaluation("FaissAndBM25EnsembleRetriever_Phi4", retriever, generator)


if __name__ == "__main__":
    evaluate_faiss_bm25_phi4()
