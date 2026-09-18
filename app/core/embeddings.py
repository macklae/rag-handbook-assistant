"""Step 4 of the RAG pipeline: turn text into vectors via the OpenAI API.

Batched because the embeddings endpoint accepts a list, and one request per
chunk would be slow and expensive on a 233-chunk document.
"""

from typing import List

from openai import OpenAI

from app.config import get_settings


class EmbeddingClient:
    def __init__(self, client: OpenAI | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings, preserving input order."""
        if not texts:
            return []

        batch_size = self._settings.embedding_batch_size
        vectors: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            # The API does not guarantee ordering, so sort by index.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
