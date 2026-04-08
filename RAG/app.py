from pathlib import Path
from uuid import uuid4

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

import RAG.config as config
from .schema import ResponseWithImages

from .domain.model_chat_service import ModelChatService
from .retrivers import EnsembleRetriever
from .domain.context_service import ContextService
from .retrivers._faiss_chunk_repository import (
    FaissChunkRepository,
)
from .infrastructure.ollama_llm_chat_adapter import OllamaLLMChatAdapter


chat_service: ModelChatService


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global chat_service
    chat_service = ModelChatService(
        context_service=ContextService(
            notes_repository=EnsembleRetriever(),
            photos_repository=FaissChunkRepository(
                filename=Path(config.PHOTO_CONTEXT_CACHE))
        ),
        generator=OllamaLLMChatAdapter(),
    )
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=config.COOKIE_SECRET_KEY,
    session_cookie="chat_session",
    max_age=60 * 60 * 24,  # 1 day
)

app.mount("/api/v1/photos", StaticFiles(directory=config.PHOTO_DIR), name="photos")


@app.post("/api/v1/query")
def query(query: str, request: Request) -> ResponseWithImages:
    """
    Endpoint to handle queries.
    """
    # Here you would typically process the query and return a response.
    # For now, we will just return the query as a placeholder.
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid4())

    response, image_ids = chat_service.ask(
        query, request.session["session_id"]
    )
    print(f"Session ID: {request.session['session_id']}")
    return ResponseWithImages(text=response, image_ids=image_ids)


@app.get("/api/v1/history")
def get_history(request: Request) -> list[str]:
    """
    Endpoint to retrieve message history for a given session.
    """
    if "session_id" not in request.session:
        return []

    return chat_service.get_history(request.session["session_id"])


@app.delete("/api/v1/history")
def clear_history(request: Request) -> None:
    """
    Endpoint to clear message history for a given session.
    """
    if "session_id" in request.session:
        chat_service.clear_history(request.session["session_id"])
        del request.session["session_id"]
