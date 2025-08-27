from .base import DocumentStore
from .elastic import ElasticsearchDocumentStore

__all__ = [
    "DocumentStore",
    "ElasticsearchDocumentStore"
]
