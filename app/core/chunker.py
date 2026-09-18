"""Step 3 of the RAG pipeline: split page text into overlapping chunks.

Character-based chunking, same as the original notebook. The one change is
that chunking happens per page rather than across the whole document, so
every chunk carries the page it came from and the UI can cite it.

Trade-off accepted: a sentence spanning a page break is split. For a
handbook of discrete policy sections that is a fair price for citations.
"""

from dataclasses import dataclass
from typing import List

from app.core.pdf_reader import Page


@dataclass
class Chunk:
    id: str
    text: str
    page: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a single string into overlapping windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step

    return chunks


def chunk_pages(pages: List[Page], chunk_size: int, overlap: int) -> List[Chunk]:
    """Turn a list of pages into a flat list of chunks with page metadata."""
    chunks: List[Chunk] = []
    counter = 0

    for page in pages:
        for piece in chunk_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(id=f"chunk-{counter}", text=piece, page=page.number))
            counter += 1

    return chunks
