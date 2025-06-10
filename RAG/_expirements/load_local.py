import glob
import os
from langchain.schema import Document


def find_md_file_paths(root_path: str) -> list[str]:
    glob_path = os.path.join(root_path, "**", "*.md")
    return glob.glob(glob_path, recursive=True)


def load_doc(doc_path: str) -> Document:
    with open(doc_path, "r") as doc:
        return Document(page_content=doc.read(), metadata={"url": doc_path})


def load_documents(root_path: str) -> list[Document]:
    paths = find_md_file_paths(root_path)
    return [load_doc(path) for path in paths]


if __name__ == "__main__":
    docs = load_documents("RAG/_expirements/hse_conspects_course1/")
    print(docs[0].page_content)
