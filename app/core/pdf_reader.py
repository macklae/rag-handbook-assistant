"""Step 2 of the RAG pipeline: extract text from a PDF, one entry per page.

Keeping pages separate (rather than joining the whole document into one
string) is what makes page-level citations possible later on.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Page:
    number: int  # 1-based, matches what a human sees in a PDF viewer
    text: str


def read_pdf(path: str | Path) -> List[Page]:
    """Return non-empty pages from the PDF at `path`."""
    reader = PdfReader(str(path))
    pages: List[Page] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(number=page_number, text=text))

    return pages


def total_characters(pages: List[Page]) -> int:
    return sum(len(page.text) for page in pages)
