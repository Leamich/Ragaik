import asyncio
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

import QA.config as config
from .schema import ResponseWithImages

from .graph import graph, get_image_ids, get_chat_history, ainvoke
from .connectors import checkpointer


app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=config.COOKIE_SECRET_KEY,
    session_cookie="chat_session",
    max_age=60 * 60 * 24,  # 1 day
)

app.mount("/api/v1/photos", StaticFiles(directory=config.PHOTO_DIR), name="photos")


@app.post("/api/v1/query")
async def query(query: str, request: Request) -> ResponseWithImages:
    """
    Endpoint to handle queries.
    """
    # Here you would typically process the query and return a response.
    # For now, we will just return the query as a placeholder.
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid4())

    response, image_ids = asyncio.gather(ainvoke(query, graph, request.session["session_id"]),
                                         get_image_ids(graph, request.session["session_id"]))
    print(f"Session ID: {request.session['session_id']}")
    return ResponseWithImages(text=response, image_ids=image_ids)


@app.get("/api/v1/history")
async def get_history(request: Request) -> list[str]:
    """
    Endpoint to retrieve message history for a given session.
    """
    if "session_id" not in request.session:
        return []
    # TODO: what type do we need?
    return await get_chat_history(graph, request.session["session_id"])


@app.delete("/api/v1/history")
def clear_history(request: Request) -> None:
    """
    Endpoint to clear message history for a given session.
    """
    if "session_id" in request.session:
        await checkpointer.adelete_thread(thread_id=request.session["session_id"]) #TODO: maybe doesn't work
