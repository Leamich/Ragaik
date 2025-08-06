from typing import TypedDict
from elasticsearch import AsyncElasticsearch
from langgraph.graph import StateGraph

from rag.domain.document import Dcument
from rag.infrastructure.retriever.elastic import AsyncElasticsearchRetriever


# instantiate once (could be injected)
#
#
class GrpahState(TypedDict):
    query: str
    docs: list[Dcument]


_elasetic = AsyncElasticsearch("dummy")
_retriever: AsyncElasticsearchRetriever = AsyncElasticsearchRetriever(
    _elasetic, index="dummy"
)


async def retrieve_node(state):
    """
    LangGraph node: reads state['query'], writes state['docs'].
    """
    docs = await _retriever.retrieve(state["query"])
    state["docs"] = docs
    return state


def build_search_graph():
    graph = StateGraph(state_schema=GrpahState)
    graph.add_node("retrieve", retrieve_node)
    # add more nodes here...
    return graph


# if needed elsewhere (e.g. to close client)
async def shutdown_retriever():
    await _elasetic.close()
