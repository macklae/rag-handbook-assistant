"""HTTP layer. Thin: validation and error mapping only, no RAG logic."""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from app.services.dependencies import get_vector_store
from app.services.ingest_service import ingest_pdf
from app.services.rag_service import answer_question

router = APIRouter(prefix="/api")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    try:
        count = get_vector_store().count()
        ready = True
    except Exception:
        count = 0
        ready = False

    return HealthResponse(
        status="ok" if ready else "degraded",
        openai_key_present=bool(settings.openai_api_key),
        vector_store_ready=ready,
        chunk_count=count,
    )


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    settings = get_settings()
    store = get_vector_store()
    return StatsResponse(
        collection_name=store.name,
        chunk_count=store.count(),
        sources=store.sources(),
        embedding_model=settings.embedding_model,
        generation_model=settings.generation_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    settings = get_settings()

    if not settings.openai_api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not set. Check your .env file.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted.")

    destination = settings.uploads_dir / Path(file.filename).name
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle, length=1024 * 1024)

    if destination.stat().st_size > MAX_UPLOAD_BYTES:
        destination.unlink(missing_ok=True)
        raise HTTPException(413, "File is larger than the 25 MB limit.")

    try:
        summary = ingest_pdf(destination)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(500, f"Ingestion failed: {error}") from error

    return IngestResponse(**summary)


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    settings = get_settings()

    if not settings.openai_api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not set. Check your .env file.")
    if get_vector_store().count() == 0:
        raise HTTPException(409, "No document has been ingested yet. Upload a PDF first.")

    try:
        result = answer_question(payload.question, top_k=payload.top_k)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(502, f"Query failed: {error}") from error

    return QueryResponse(**result)


@router.delete("/reset")
def reset() -> dict:
    removed = get_vector_store().clear()
    return {"removed": removed, "chunk_count": get_vector_store().count()}
