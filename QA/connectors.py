from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import (
    MAIN_MODEL_KWARGS, HYDE_MODEL_KWARGS, EMBEDDINGS_MODEL_KWARGS, REDIS_API_URL
)
from langgraph.checkpoint.redis import AsyncRedisSaver

embedding_model = OpenAIEmbeddings(**EMBEDDINGS_MODEL_KWARGS)
main_model = ChatOpenAI(**MAIN_MODEL_KWARGS)
hyde_model = ChatOpenAI(**HYDE_MODEL_KWARGS)

checkpointer = AsyncRedisSaver.from_conn_string(REDIS_API_URL)
checkpointer.setup() #TODO: maybe doesn't work