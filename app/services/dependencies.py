"""Single place where the pipeline objects are constructed and shared.

Chroma's PersistentClient and the OpenAI client are both expensive to create
per request, so they are built once and reused.
"""

from functools import lru_cache

from app.core.embeddings import EmbeddingClient
from app.core.generator import Generator
from app.core.retriever import Retriever
from app.core.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache
def get_generator() -> Generator:
    return Generator()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_vector_store(), get_embedding_client())
