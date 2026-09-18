"""Ingestion: PDF in, embedded chunks in the vector store."""

from pathlib import Path
from typing import Dict

from app.config import get_settings
from app.core.chunker import chunk_pages
from app.core.pdf_reader import read_pdf, total_characters
from app.services.dependencies import get_embedding_client, get_vector_store


def ingest_pdf(path: Path, replace: bool = True) -> Dict[str, int | str]:
    """Read, chunk, embed and store a PDF. Returns a summary."""
    settings = get_settings()
    store = get_vector_store()
    embedder = get_embedding_client()

    pages = read_pdf(path)
    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be a scan, which needs OCR."
        )

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)

    # Replacing by default: re-uploading the same handbook should not create
    # duplicate chunks that then compete with each other at retrieval time.
    if replace:
        store.clear()

    embeddings = embedder.embed([chunk.text for chunk in chunks])
    store.add(chunks, embeddings, source=path.name)

    return {
        "filename": path.name,
        "pages": len(pages),
        "characters": total_characters(pages),
        "chunks": len(chunks),
        "collection_count": store.count(),
    }
