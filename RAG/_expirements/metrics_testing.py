import pandas as pd
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    ResponseGroundedness,
)
from ragas.evaluation import evaluate
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..domain.port.generator import RussianPhi4Generator
from .load_local import load_documents


def run_ragas_evaluation(system_name: str, retriever, generator):
    df = pd.read_csv("test.tsv", sep="\t", names=["question", "answer"])
    ragas_inputs = []

    for _, row in df.iterrows():
        query = row["question"]
        gt_answer = row["answer"]

        contexts = retriever.query(query)
        generated = generator.generate(query, contexts)

        ragas_inputs.append(
            {
                "question": query,
                "answer": generated,
                "contexts": contexts,
                "ground_truth": gt_answer,
            }
        )

    results = evaluate(
        ragas_inputs,
        metrics=[
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
            ResponseGroundedness,
        ],
        raise_exceptions=False,
    )

    mean_scores = results.mean()
    mean_scores["system"] = system_name

    try:
        existing_df = pd.read_csv("result.tsv", sep="\t")
        result_df = pd.concat(
            [existing_df, mean_scores.to_frame().T], ignore_index=True
        )
    except FileNotFoundError:
        result_df = mean_scores.to_frame().T

    result_df.to_csv("result.tsv", sep="\t", index=False)

    print(f"\n--- RAGAS Evaluation Summary: {system_name} ---")
    print(mean_scores)


def evaluate_faiss_bm25_phi4():
    retriever = FaissAndBM25EnsembleRetriever()
    generator = RussianPhi4Generator()

    start_path = "RAG/_expirements/hse_conspects_course1/"
    documents = load_documents(start_path)
    retriever.add_batch(documents)

    run_ragas_evaluation("FaissAndBM25EnsembleRetriever + Phi4", retriever, generator)


if __name__ == "__main__":
    evaluate_faiss_bm25_phi4()
