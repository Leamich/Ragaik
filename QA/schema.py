import operator
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import MessagesState
from typing import TypeAlias
from langchain_core.documents import Document


class ResponseWithImages(BaseModel):
    text: str
    image_ids: set[str]


Context: TypeAlias = set[Document]

class State(MessagesState):
    hydified_query: str
    context: Annotated[Context, operator.add]
    image_ids: Annotated[set[str], operator.add]