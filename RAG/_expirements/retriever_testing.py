import requests
from langchain.schema import Document
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever

REPO_OWNER = "Leamich"
REPO_NAME = "hse_conspects_course1"
API_BASE = "https://api.github.com/repos"
RAW_URL_PREFIX = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/master"

HEADERS = {
    "Accept": "application/vnd.github.v3+json"
}

def get_all_md_files(path):
    url = f"{API_BASE}/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Ошибка API: {url} ({response.status_code})")
        return []

    md_files = []
    for item in response.json():
        if item['type'] == 'file' and item['name'].endswith('.md'):
            raw_url = f"{RAW_URL_PREFIX}/{item['path']}"
            md_files.append(raw_url)
        elif item['type'] == 'dir':
            md_files.extend(get_all_md_files(item['path']))
    return md_files

def download_and_create_documents(md_urls):
    documents = []
    for url in md_urls:
        print(f"Загрузка: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            text = response.text
            doc = Document(page_content=text, metadata={"url": url})
            documents.append(doc)
        else:
            print(f"Не удалось загрузить {url}")
    return documents

if __name__ == "__main__":
    retr = FaissAndBM25EnsembleRetriever()
    start_path = "hse/ma/conspects"
    md_urls = get_all_md_files(start_path)
    print(f"Найдено {len(md_urls)} md файлов.")
    documents = download_and_create_documents(md_urls)
    print(f"Создано {len(documents)} документов.")

    retr.add_batch(documents)
    retr.query("Что такое бинарные отношения? Какие у них бывают свойства?")

    
