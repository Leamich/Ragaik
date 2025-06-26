from langchain_core import embeddings
import pandas as pd
from ragas.metrics import (
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from ragas.utils import Dataset
from transformers.models.gptj.modeling_gptj import get_embed_positions
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..domain.port.generator import RussianPhi4Generator
from .load_local import load_documents

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


def run_ragas_evaluation(
    system_name: str, retriever: FaissAndBM25EnsembleRetriever, generator
):
    critic_llm = OllamaLLM(model="phi4")
    wrapped_critic = LangchainLLMWrapper(critic_llm)

    embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embedder_wrapped = LangchainEmbeddingsWrapper(embedder)

    run_config = RunConfig(max_workers=1, timeout=500)

    df = pd.read_csv("test.tsv", sep="\t", names=["question", "answer"])
    datasamples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for _, row in df.iterrows():
        query = str(row["question"])
        gt_answer = str(row["answer"])

        contexts = retriever.query(query)
        if contexts is None:
            contexts = []
        generated = generator.generate(query, contexts)

        datasamples["question"].append(query)
        datasamples["answer"].append(generated)
        datasamples["contexts"].append([doc.page_content for doc in contexts])
        datasamples["ground_truth"].append(gt_answer)

    dataset = Dataset.from_dict(datasamples)
    results: pd.DataFrame = evaluate(
        dataset,
        metrics=[
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            Faithfulness()
        ],
        raise_exceptions=False,
        llm=wrapped_critic,
        embeddings=embedder_wrapped,
        run_config=run_config,
    ).to_pandas()

    results.to_pickle("res.pkl")


def evaluate_faiss_bm25_phi4():
    retriever = FaissAndBM25EnsembleRetriever()
    generator = RussianPhi4Generator()

    start_path = "RAG/_expirements/hse_conspects_course1/"
    documents = load_documents(start_path)
    retriever.add_batch(documents)

    run_ragas_evaluation("FaissAndBM25EnsembleRetriever + Phi4", retriever, generator)


if __name__ == "__main__":
    evaluate_faiss_bm25_phi4()
