"""Request and response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    page: int
    text: str
    distance: float = Field(description="Chroma L2 distance. Lower is more similar.")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(
        description="False when no chunk passed the distance threshold, "
        "meaning the answer was refused rather than generated."
    )
    chunks: List[RetrievedChunk]
    generation_model: str
    embedding_model: str
    latency_ms: int


class IngestResponse(BaseModel):
    filename: str
    pages: int
    characters: int
    chunks: int
    collection_count: int


class StatsResponse(BaseModel):
    collection_name: str
    chunk_count: int
    sources: List[str]
    embedding_model: str
    generation_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


class HealthResponse(BaseModel):
    status: str
    openai_key_present: bool
    vector_store_ready: bool
    chunk_count: int
