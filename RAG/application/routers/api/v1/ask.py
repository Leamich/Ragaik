from typing import Annotated
from fastapi import APIRouter, Request
from uuid import uuid4

from .....application.schema.ask_schema import QuerySchema, ResponseSchema
from .....domain.service import RAGService
from fastapi.params import Depends
from ....deps import get_rag_service

ask_router = APIRouter()


@ask_router.post("/query")
def query(
    query_schema: QuerySchema,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    request: Request,
) -> ResponseSchema:
    """
    Endpoint to handle queries.
    """
    # Here you would typically process the query and return a response.
    # For now, we will just return the query as a placeholder.
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid4())

    response, _ = rag_service.ask(query_schema.query, request.session["session_id"])
    print(f"Session ID: {request.session['session_id']}")
    return ResponseSchema(response=response)

@ask_router.get("/history")
def get_history(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    request: Request,
) -> list[str]:
    """
    Endpoint to retrieve message history for a given session.
    """
    if "session_id" not in request.session:
        return []

    return rag_service.get_history(request.session["session_id"])

@ask_router.delete("/history")
def clear_history(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    request: Request,
) -> None:
    """
    Endpoint to clear message history for a given session.
    """
    if "session_id" in request.session:
        rag_service.clear_history(request.session["session_id"])
        del request.session["session_id"]
