from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from .load_local import load_documents


if __name__ == "__main__":
    retr = FaissAndBM25EnsembleRetriever()
    start_path = "RAG/tests/hse_conspects_course1/"
    documents = load_documents(start_path)
    for doc in documents:
        if doc is None:
            print("SHIT")
    print(f"Создано {len(documents)} документов.")

    retr.add_batch(documents)
    print(retr.query("Что такое бинарные отношения? Какие у них бывают свойства?"))
