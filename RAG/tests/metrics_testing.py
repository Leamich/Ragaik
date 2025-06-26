import os
import pandas as pd
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    Faithfulness
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

from tqdm import tqdm



def run_ragas_evaluation(
    system_name: str, retriever: FaissAndBM25EnsembleRetriever, generator
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

    if not os.path.exists("answers.pkl"):
        for question in tqdm(questions, desc="Generating answers"):
            contexts = retriever.query(question)
            generated = generator.generate(question, contexts)

            datasamples["question"].append(question)
            datasamples["answer"].append(generated)
            datasamples["contexts"].append([doc.page_content for doc in contexts])
            datasamples["referense"].append([""])

        answers = pd.DataFrame.from_dict(datasamples)
        answers.to_pickle("answers.pkl")
        dataset = Dataset.from_dict(datasamples)

    else:
        answers = pd.read_pickle("answers.pkl")
        dataset = Dataset.from_pandas(answers)

    dataset = Dataset.from_dict(datasamples)
    pd.DataFrame(datasamples).to_excel(f"{system_name}_answers.xlsx")

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
    generator = RussianPhi4LLMChatAdapter()

    print("INFO: loading documents")
    start_path = "RAG/tests/hse_conspects_course1/"
    documents = load_documents(start_path)
    print("INFO: loaded", len(documents), "documents")

    print("INFO: adding documents to retriever")
    retriever.add_batch(documents)

    print("INFO: running evaluation")
    run_ragas_evaluation("FaissAndBM25EnsembleRetriever_Phi4", retriever, generator)


if __name__ == "__main__":
    evaluate_faiss_bm25_phi4()
