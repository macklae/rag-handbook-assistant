"""Step 9 of the RAG pipeline: assemble the prompt from question + context."""

from typing import Any, Dict, List

SYSTEM_RULES = (
    "You are an assistant answering questions about a company employee handbook.\n"
    "Answer using ONLY the retrieved context below.\n"
    "If the answer is not present in the context, say that the information is "
    "not available in the provided document. Do not use outside knowledge.\n"
    "Cite the page number in brackets after each fact, for example [p. 22]."
)

REFUSAL = (
    "I could not find anything relevant to that question in the uploaded "
    "document, so I have nothing to answer from."
)

CHUNK_SEPARATOR = "\n\n--- Retrieved Chunk ---\n\n"


def build_context(chunks: List[Dict[str, Any]]) -> str:
    return CHUNK_SEPARATOR.join(
        f"[page {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks)
    return (
        f"{SYSTEM_RULES}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Question: {question}"
    ).strip()
