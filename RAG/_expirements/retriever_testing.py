import requests
import os
import hashlib
import asyncio
import aiohttp
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

    def get_local_path(url):
        hash_object = hashlib.md5(url.encode())
        filename = f"{hash_object.hexdigest()}.md"
        return os.path.join("cached_documents", filename)

    async def download_document(url):
        local_path = get_local_path(url)
        if os.path.exists(local_path):
            print(f"Загрузка из локального кэша: {local_path}")
            with open(local_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            print(f"Загрузка: {url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        os.makedirs("cached_documents", exist_ok=True)
                        with open(local_path, "w", encoding="utf-8") as file:
                            file.write(text)
                    else:
                        print(f"Не удалось загрузить {url}")
                        return None
        return Document(page_content=text, metadata={"url": url})

    documents = []
    async def download_all_documents(urls):
        tasks = [download_document(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    documents = asyncio.run(download_all_documents(md_urls))

    return documents

if __name__ == "__main__":
    retr = FaissAndBM25EnsembleRetriever()
    start_path = "hse/ma/conspects"
    md_urls = get_all_md_files(start_path)
    print(f"Найдено {len(md_urls)} md файлов.")
    documents = download_and_create_documents(md_urls)
    print(f"Создано {len(documents)} документов.")

    retr.add_batch(documents)
    print(retr.query("Что такое бинарные отношения? Какие у них бывают свойства?"))
