from fastapi import APIRouter
from .ask import ask_router


api_v1_router = APIRouter()
api_v1_router.include_router(ask_router, prefix="/ask", tags=["ask"])
