"""Query orchestration: retrieve, build the prompt, generate, return sources."""

import time
from typing import Any, Dict

from app.config import get_settings
from app.core.prompt import REFUSAL, build_prompt
from app.services.dependencies import get_generator, get_retriever


def answer_question(question: str, top_k: int | None = None) -> Dict[str, Any]:
    settings = get_settings()
    started = time.perf_counter()

    retriever = get_retriever()
    chunks = retriever.retrieve(question, top_k=top_k)

    if not chunks:
        # Nothing cleared the distance threshold. Refuse rather than let the
        # model answer from its own knowledge and sound authoritative.
        return {
            "question": question,
            "answer": REFUSAL,
            "grounded": False,
            "chunks": [],
            "generation_model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    prompt = build_prompt(question, chunks)
    answer = get_generator().generate(prompt)

    return {
        "question": question,
        "answer": answer,
        "grounded": True,
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": round(chunk["distance"], 4),
            }
            for chunk in chunks
        ],
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
