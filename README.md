# Enterprise Knowledge Assistant

RAG over a PDF employee handbook. Ask a question in natural language, get an answer grounded only in the retrieved text, with page citations and the retrieved chunks shown alongside.

Built from a Colab notebook into a modular FastAPI service. No LangChain: every step of the pipeline is explicit, which is the point.

```
PDF  ->  pypdf  ->  chunking  ->  text-embedding-3-small  ->  ChromaDB
                                                                 |
answer  <-  Responses API  <-  prompt + context  <-  retrieval  <-'
```

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your OPENAI_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000, upload a PDF, ask a question.

API docs at http://localhost:8000/docs. In VS Code, press F5 and pick "FastAPI: uvicorn --reload".

---

## Project structure

```
rag-handbook-assistant/
|-- app/
|   |-- main.py                  FastAPI app, static mount
|   |-- config.py                Settings from .env, one cached instance
|   |-- schemas.py               Request and response contracts
|   |-- api/
|   |   `-- routes.py            HTTP layer only: validation, error mapping
|   |-- core/                    One module per pipeline step
|   |   |-- pdf_reader.py        Extract text, keep pages separate
|   |   |-- chunker.py           Character chunking, per page
|   |   |-- embeddings.py        Batched OpenAI embeddings
|   |   |-- vector_store.py      ChromaDB persistence and search
|   |   |-- retriever.py         Query embedding + distance filter
|   |   |-- prompt.py            Prompt assembly and refusal text
|   |   `-- generator.py         Responses API call
|   `-- services/
|       |-- dependencies.py      Shared singletons
|       |-- ingest_service.py    Orchestrates ingestion
|       `-- rag_service.py       Orchestrates a query
|-- eval/
|   |-- golden_set.json          Questions with expected pages
|   |-- metrics.py               Hit rate, MRR
|   `-- run_eval.py              CLI harness
|-- static/                      Frontend: index.html, styles.css, app.js
|-- tests/                       pytest
`-- data/                        Uploads and the Chroma database (gitignored)
```

The layering is deliberate. `core/` knows nothing about HTTP. `api/` knows nothing about embeddings. Swapping ChromaDB for pgvector, or OpenAI for Bedrock, means editing one file in `core/` and nothing else.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/ingest` | Upload a PDF, chunk, embed, store. Replaces the existing index |
| `POST` | `/api/query` | Ask a question. Returns answer, chunks, pages, distances, latency |
| `GET` | `/api/stats` | Chunk count, sources, active models and chunking config |
| `GET` | `/api/health` | Key present, vector store reachable, chunk count |
| `DELETE` | `/api/reset` | Empty the collection |

---

## Evaluation

```bash
python -m eval.run_eval                  # retrieval metrics, no LLM calls
python -m eval.run_eval --with-answers   # adds generation and refusal checks
python -m eval.run_eval --top-k 5        # sweep K
```

Two metrics, because they answer different questions:

- **Hit rate** asks whether the right content reached the model at all.
- **MRR** asks whether it reached the model *early*. Context windows are ordered and budgeted, so a correct chunk at rank 8 is worth less than the same chunk at rank 1.

Retrieval is scored without calling the LLM. When an answer is wrong, the first thing to establish is whether retrieval failed or generation failed, and an end-to-end score hides that.

`--with-answers` adds two more checks: a keyword pass on the expected answer, and a **refusal rate** over deliberately out-of-scope questions. A RAG system that answers "who won the 2019 cricket world cup" has a broken guardrail, and only the refusal test catches it.

Results are written to `eval/results/` together with the config that produced them, so runs stay comparable after a chunk size or K change.

**Before the first run:** open `eval/golden_set.json` and correct `expected_pages` against the actual PDF. The page numbers shipped here are estimates from the table of contents. Retrieval metrics are only as good as the labels.

---

## Product decisions

**Chunk per page, not per document.** The notebook joined all 57 pages into one string before chunking, which made page citations impossible. Chunking within each page means every chunk carries its page number and the UI can cite it. The cost is that a sentence spanning a page break gets split. For a handbook of discrete policy sections that is worth paying. For a flowing narrative document it would not be.

**A distance threshold, not just top K.** Chroma always returns the nearest chunks, even when the nearest chunk is irrelevant. Without a threshold an off-topic question still arrives at the model wrapped in confident-looking context. `MAX_DISTANCE` drops chunks that are merely least-bad, and the service refuses instead of generating. The `grounded` flag in the response says which path was taken.

**Show the retrieved chunks in the UI.** Users trust an answer they can check. Exposing chunk ID, page and distance also turns the frontend into a debugging tool: when an answer is wrong, the chunks show immediately whether retrieval or generation was at fault.

**Ingestion replaces rather than appends.** Re-uploading the same handbook would otherwise create duplicate chunks that compete with each other at retrieval time and crowd out the top K.

**Retrieval evaluated separately from generation.** See above.

---

## Configuration

Everything lives in `.env`, with defaults matching the original notebook:

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dimensions |
| `GENERATION_MODEL` | `gpt-4.1-mini` | Via the Responses API |
| `CHUNK_SIZE` | `500` | Characters |
| `CHUNK_OVERLAP` | `50` | Characters |
| `TOP_K` | `3` | Chunks passed to the model |
| `MAX_DISTANCE` | `1.2` | Above this, a chunk is dropped |

Changing chunk size or overlap requires re-ingesting, since existing embeddings were built under the old settings.

---

## Known limitations

- Character chunking splits mid-sentence. Token-aware or heading-aware chunking would retrieve better.
- No reranker. A cross-encoder over the top 20 would lift MRR.
- Single collection, single document. Multi-document needs a source filter on the query.
- No conversation memory. Each question is independent.
- Scanned PDFs return no text. OCR is not wired in.
- `MAX_DISTANCE` was set by observation, not tuned. Sweep it against the golden set.

---

## Tests

```bash
pytest
```

Covers chunking boundaries, overlap validation, page metadata and the retrieval metrics. No network calls.
