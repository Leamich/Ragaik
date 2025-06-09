from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from .load_documents import get_all_md_files, download_and_create_documents


if __name__ == "__main__":
    retr = FaissAndBM25EnsembleRetriever()
    start_path = "hse/ma/conspects"
    md_urls = get_all_md_files(start_path)
    print(f"Найдено {len(md_urls)} md файлов.")
    documents = download_and_create_documents(md_urls)
    print(f"Создано {len(documents)} документов.")

    retr.add_batch(documents)
    print(retr.query("Что такое бинарные отношения? Какие у них бывают свойства?"))
