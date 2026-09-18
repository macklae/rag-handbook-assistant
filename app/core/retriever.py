"""Steps 7 and 8 of the RAG pipeline: embed the query, search the store.

The distance filter is the guardrail. Chroma always returns the nearest
chunks, even when the nearest chunk is irrelevant, so without a threshold an
off-topic question still reaches the LLM with confident-looking context.
"""

from typing import Any, Dict, List

from app.config import get_settings
from app.core.embeddings import EmbeddingClient
from app.core.vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingClient) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = get_settings()

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or self._settings.top_k
        query_vector = self._embedder.embed_one(question)
        hits = self._store.query(query_vector, top_k=k)
        return [h for h in hits if h["distance"] <= self._settings.max_distance]

    def retrieve_unfiltered(
        self, question: str, top_k: int | None = None
    ) -> List[Dict[str, Any]]:
        """Used by the evaluation harness, which needs raw ranking."""
        k = top_k or self._settings.top_k
        query_vector = self._embedder.embed_one(question)
        return self._store.query(query_vector, top_k=k)
