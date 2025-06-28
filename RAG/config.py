import os
from typing import Final

COOKIE_SECRET_KEY: Final = os.getenv("COOKIE_SECRET_KEY", "supersecretkey")
OLLAMA_MODEL_NAME: Final = os.getenv("OLLAMA_MODEL_NAME", "phi4")
OLLAMA_API_URL: Final = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
REDIS_API_URL: Final = os.getenv("REDIS_API_URL", "redis://localhost:6379")




FAISS_CACHE_DIR: Final = os.getenv("FAISS_CACHE_DIR", "./faiss_cache")
BM25_CACHE_FILE: Final = os.getenv("BM25_CACHE_FILE", "./bm25_cache.pkl")
PHOTO_DIR: Final = os.getenv("PHOTO_DIR", "./resources/photos/dump_all_AIK")
PHOT_TEXT_CONTENT_FILE: Final = os.getenv("PHOT_TEXT_CONTENT_FILE", "resources/photos/phot_text_content.csv") 
PHOTO_CONTEXT_CACHE: Final = os.getenv("PHOTO_CONTEXT", "./faiss_photo_cache")
NOTES_START_DIR: Final = os.getenv("NOTES_START_FILE", "RAG/tests/hse_conspects_course1/") 