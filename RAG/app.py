from fastapi import FastAPI
from .application.routers.api.v1.api_v1 import api_v1_router

from starlette.middleware.sessions import SessionMiddleware
import RAG.config as config
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=config.COOKIE_SECRET_KEY,
    session_cookie="chat_session",
    max_age=60 * 60 * 24,  # 1 day
)


app.include_router(api_v1_router, prefix="/api/v1", tags=["v1"])
