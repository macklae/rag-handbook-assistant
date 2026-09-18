"""Step 5 of the RAG pipeline: persist chunks and embeddings in ChromaDB.

The vector store does not generate answers. It stores embeddings and returns
the nearest chunks for a query vector.
"""

from typing import Any, Dict, List

import chromadb

from app.config import get_settings
from app.core.chunker import Chunk


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name
        )

    @property
    def name(self) -> str:
        return self._settings.collection_name

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> int:
        """Remove every document. Returns how many were removed."""
        existing = self._collection.get()
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def add(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        source: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return

        self._collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[{"source": source, "page": chunk.page} for chunk in chunks],
        )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Return the top_k nearest chunks as plain dicts."""
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding], n_results=min(top_k, self.count())
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        return [
            {
                "chunk_id": chunk_id,
                "text": document,
                "distance": float(distance),
                "page": int(metadata.get("page", 0)),
                "source": metadata.get("source", ""),
            }
            for chunk_id, document, distance, metadata in zip(
                ids, documents, distances, metadatas
            )
        ]

    def sources(self) -> List[str]:
        """Distinct source filenames currently in the collection."""
        existing = self._collection.get()
        metadatas = existing.get("metadatas") or []
        found = {m.get("source") for m in metadatas if m and m.get("source")}
        return sorted(found)
