from typing import Annotated
from fastapi import APIRouter

from .....application.schema.ask_schema import QuerySchema, ResponseSchema
from .....domain.service import RAGService
from fastapi.params import Depends
from ....deps import get_rag_service

ask_router = APIRouter()

@ask_router.post("/query")
def query(query: QuerySchema, rag_service: Annotated[RAGService, Depends(get_rag_service)]) -> ResponseSchema:
    """
    Endpoint to handle queries.
    """
    # Here you would typically process the query and return a response.
    # For now, we will just return the query as a placeholder.
    return ResponseSchema(response=rag_service.ask(query.query))
