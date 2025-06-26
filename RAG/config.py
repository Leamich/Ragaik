import os
from typing import Final

COOKIE_SECRET_KEY: Final = os.getenv("COOKIE_SECRET_KEY", "huy")
OLLAMA_MODEL_NAME: Final = os.getenv("OLLAMA_MODEL_NAME", "phi3.5")
OLLAMA_API_URL: Final = os.getenv("OLLAMA_API_URL", "http://localhost:11434")