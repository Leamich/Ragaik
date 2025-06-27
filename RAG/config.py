import os
from typing import Final

COOKIE_SECRET_KEY: Final = os.getenv("COOKIE_SECRET_KEY", "huy")
OLLAMA_MODEL_NAME: Final = os.getenv("OLLAMA_MODEL_NAME", "phi4")
OLLAMA_API_URL: Final = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

FAISS_CACHE_DIR: Final = os.getenv("FAISS_CACHE_DIR", "./faiss_cache")
BM25_CACHE_FILE: Final = os.getenv("BM25_CACHE_FILE", "./bm25_cache.pkl")
PHOTO_DIR: Final = os.getenv("PHOTO_DIR", "./resources/photos/files")
PHOTO_CONTEXT_CACHE: Final = os.getenv("PHOTO_CONTEXT", "./faiss_photo_cache")
