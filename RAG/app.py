from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

import RAG.config as config

from .application.routers.api.v1.api_v1 import api_v1_router

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=config.COOKIE_SECRET_KEY,
    session_cookie="chat_session",
    max_age=60 * 60 * 24,  # 1 day
)


app.include_router(api_v1_router, prefix="/api/v1", tags=["v1"])
app.mount("/api/v1/photos", StaticFiles(directory=config.PHOTO_DIR), name="photos")
