"""FastAPI entry point.

Run locally:
    uvicorn app.main:app --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app.api.routes import router  # noqa: E402  (must follow load_dotenv)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG over a PDF handbook: pypdf, OpenAI embeddings, ChromaDB, Responses API.",
    version="1.0.0",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
