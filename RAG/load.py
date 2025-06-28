import glob
import os
import pandas as pd
from langchain.schema import Document
from pathlib import Path


def find_md_file_paths(root_path: Path) -> list[Path]:
    glob_path = os.path.join(root_path, "**", "*.md")
    return [Path(path) for path in glob.glob(glob_path, recursive=True)]


def load_doc(doc_path: Path) -> Document:
    with open(doc_path, "r") as doc:
        return Document(page_content=doc.read(), metadata={"url": doc_path})


def load_documents(root_path: Path) -> list[Document]:
    paths = find_md_file_paths(root_path)
    return [load_doc(path) for path in paths]


def load_photo_docs(photo_content_mapping_path: Path) -> list[Document]:
    data = pd.read_csv(photo_content_mapping_path, header=0)
    return data.apply(map_to_document, axis=1).tolist()

def map_to_document(row: pd.Series) -> Document:
    return Document(
        page_content=str(row["text"]),
        metadata={
            "image_id": row["image_id"],
        },
    )

