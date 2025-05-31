from fastapi import FastAPI
from .application.routers.api.v1.api_v1 import api_v1_router


app = FastAPI()
app.include_router(api_v1_router, prefix="/api/v1", tags=["v1"])
