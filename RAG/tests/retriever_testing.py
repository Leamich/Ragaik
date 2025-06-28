from pathlib import Path
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from ..load import load_documents
import RAG.config as config


if __name__ == "__main__":
    retr = FaissAndBM25EnsembleRetriever()
    start_path = Path(config.NOTES_START_DIR)
    documents = load_documents(start_path)
    for doc in documents:
        if doc is None:
            print("There is no documents")
    print(f"Создано {len(documents)} документов.")

    retr.add_batch(documents)
    print(retr.query("Что такое бинарные отношения? Какие у них бывают свойства?"))
