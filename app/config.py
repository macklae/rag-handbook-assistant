"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4.1-mini"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 3
    # Chroma returns squared L2 distance by default: lower is closer.
    # Chunks beyond this distance are treated as irrelevant and dropped.
    max_distance: float = 1.2

    # --- Storage ---
    chroma_path: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    collection_name: str = "employee_handbook"

    # --- Embedding batching ---
    embedding_batch_size: int = 100

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def uploads_dir(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached so the whole app shares one Settings instance."""
    return Settings()
