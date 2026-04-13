from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


from .config import (
    HYDE_PROMPT, MAIN_PROMPT
)
from .connectors import main_model, hyde_model
from .schema import State, Context
from .retrivers import EnsembleRetriever

#TODO: sometimes we want to remove some irrelevant context
def build_graph(checkpointer) -> CompiledStateGraph:
    retriever = EnsembleRetriever()
    async def hydify(state: State) -> dict[str, str]:
        response = await hyde_model.ainvoke([
            SystemMessage(content=HYDE_PROMPT),
            *state["messages"],
        ])

        return {'hydified_query': response.text}
    
    async def retrieve(state: State) -> dict[str, Context]:
        context = await retriever.query(state['hydified_query'])
        return {'context': context}
    
    def format_contexts(contexts: Context) -> str:
        if not contexts:
            return "Контекст отсутствует."
        return "\n\n".join(
            f"Источник {i}:\n{doc.page_content}"
            for i, doc in enumerate(contexts, start=1)
        )

    async def generate(state: State) -> dict[str, list[AIMessage]]:
        context_block = format_contexts(state["context"])

        prompt = [
            SystemMessage(MAIN_PROMPT),
            SystemMessage(f"Контекст:\n{context_block}"),
            *state["messages"]
        ]

        response = await main_model.ainvoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("hydify", hydify)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "hydify")
    builder.add_edge("hydify", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


async def get_image_ids(graph: CompiledStateGraph, session_id: str) -> set[str]:
    state = await graph.aget_state(config={"configurable": {"thread_id": session_id}})
    
    return state.values['image_ids']

async def get_chat_history(graph: CompiledStateGraph, session_id: str) -> list[dict[str, str]]:
    state = await graph.aget_state(config={"configurable": {"thread_id": session_id}})
    
    return [{"role": m.type, "content": m.content} for m in state.values["messages"]]

async def ainvoke(query, graph: CompiledStateGraph, session_id: str) -> str:
    response = await graph.ainvoke(query, config={"configurable": {"thread_id": session_id}})
    
    return response['messages'][0].text
