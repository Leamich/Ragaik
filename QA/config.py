import os

COOKIE_SECRET_KEY = os.getenv("COOKIE_SECRET_KEY", "supersecretkey")
REDIS_API_URL = os.getenv("REDIS_API_URL", "redis://localhost:6379")


FAISS_CACHE_DIR = os.getenv("FAISS_CACHE_DIR", "./faiss_cache")
BM25_CACHE_FILE = os.getenv("BM25_CACHE_FILE", "./bm25_cache.pkl")
PHOT_TEXT_CONTENT_FILE = os.getenv("PHOT_TEXT_CONTENT_FILE", "resources/photos/phot_text_content.csv") 
PHOTO_CONTEXT_CACHE = os.getenv("PHOTO_CONTEXT", "./faiss_photo_cache")
NOTES_START_DIR = os.getenv("NOTES_START_FILE", "RAG/tests/hse_conspects_course1/") 

# TODO add settings
MAIN_MODEL_KWARGS ={}
HYDE_MODEL_KWARGS ={}
EMBEDDINGS_MODEL_KWARGS = {}

# TODO: add prompts
MAIN_PROMPT = ''
HYDE_PROMPT = ''