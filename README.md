# Enterprise Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) application for querying an employee handbook using natural language.

Upload a PDF, ask questions about its contents, and receive answers grounded in the retrieved document content, with page citations and the source chunks displayed alongside the answer.

Built from a Colab prototype into a modular FastAPI application with an explicit RAG pipeline. No LangChain — each stage of the pipeline is implemented separately so that retrieval, prompting, generation, and evaluation can be inspected and tested independently.

---

## How It Works

The application follows an explicit RAG pipeline:

**PDF → PDF Text Extraction → Page-Aware Chunking → OpenAI Embeddings → ChromaDB Vector Store → Semantic Retrieval → Distance Filtering → Retrieved Context → Prompt + Context → OpenAI Responses API → Grounded Answer + Page Citations**

### Pipeline

1. **Upload**
   - User uploads an employee handbook PDF.

2. **Text Extraction**
   - PDF text is extracted page by page using `pypdf`.

3. **Chunking**
   - Extracted text is split into configurable chunks.
   - Each chunk retains its source page number.

4. **Embedding**
   - Chunks are converted into vector embeddings using OpenAI `text-embedding-3-small`.

5. **Vector Storage**
   - Embeddings and metadata are stored persistently in ChromaDB.

6. **Retrieval**
   - A user's question is embedded and compared against stored chunks.
   - The top-K most relevant chunks are retrieved.
   - A configurable distance threshold removes weak matches.

7. **Generation**
   - Retrieved chunks are passed to the OpenAI Responses API as context.
   - The model is instructed to answer using the retrieved context only.

8. **Response**
   - The frontend displays the generated answer.
   - Page citations and retrieved source chunks are shown alongside it.

---

## Features

- PDF document upload and ingestion
- Page-aware text extraction using `pypdf`
- Configurable text chunking
- Batched OpenAI embeddings
- Persistent ChromaDB vector store
- Semantic similarity retrieval
- Configurable Top-K retrieval
- Configurable similarity distance threshold
- Grounded answer generation
- Page-level citations
- Retrieved source chunks displayed in the UI
- Refusal when relevant context cannot be retrieved
- Retrieval evaluation using Hit Rate and MRR
- Optional end-to-end answer/refusal evaluation
- FastAPI REST API
- Swagger API documentation
- Simple browser-based frontend
- Pytest test suite
- Environment-based configuration
- No LangChain

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Language | Python |
| PDF Processing | pypdf |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI Responses API |
| Vector Database | ChromaDB |
| Configuration | Pydantic Settings |
| Frontend | HTML / CSS / JavaScript |
| Testing | pytest |

---

## Quick Start

### 1. Clone the repository

~~~bash
git clone https://github.com/macklae/rag-handbook-assistant.git
cd rag-handbook-assistant
~~~

### 2. Create a virtual environment

~~~bash
python3 -m venv .venv
~~~

Activate it:

**macOS / Linux**

~~~bash
source .venv/bin/activate
~~~

**Windows**

~~~powershell
.venv\Scripts\activate
~~~

### 3. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 4. Create your environment file

~~~bash
cp .env.example .env
~~~

Open `.env` and add your OpenAI API key:

~~~env
OPENAI_API_KEY=your_api_key_here
~~~

Do not commit `.env` to GitHub.

### 5. Start the application

~~~bash
python -m uvicorn app.main:app --reload
~~~

The application will be available at:

~~~text
http://localhost:8000
~~~

Swagger API documentation:

~~~text
http://localhost:8000/docs
~~~

---

## Configuration

The application is configured through environment variables.

Example `.env`:

~~~env
OPENAI_API_KEY=your_api_key_here

EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4.1-mini

CHUNK_SIZE=500
CHUNK_OVERLAP=50

TOP_K=3
MAX_DISTANCE=1.2

CHROMA_PATH=./data/chroma_db
UPLOAD_DIR=./data/uploads
COLLECTION_NAME=employee_handbook
~~~

### Key Settings

| Setting | Purpose |
|---|---|
| `CHUNK_SIZE` | Maximum size of each text chunk |
| `CHUNK_OVERLAP` | Number of overlapping characters between chunks |
| `TOP_K` | Number of candidate chunks retrieved for a query |
| `MAX_DISTANCE` | Maximum vector distance allowed for retrieved chunks |
| `EMBEDDING_MODEL` | OpenAI model used for embeddings |
| `GENERATION_MODEL` | OpenAI model used for answer generation |
| `CHROMA_PATH` | Persistent ChromaDB storage location |
| `UPLOAD_DIR` | Location for uploaded PDFs |

---

## Project Structure

~~~text
rag-handbook-assistant/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── pdf_reader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── prompt.py
│   │   └── generator.py
│   │
│   └── services/
│       ├── dependencies.py
│       ├── ingest_service.py
│       └── rag_service.py
│
├── eval/
│   ├── golden_set.json
│   ├── metrics.py
│   └── run_eval.py
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/
│   ├── test_chunker.py
│   └── test_metrics.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── run.sh
└── README.md
~~~

---

## Architecture

The application is separated into API, service, and core layers.

~~~text
Browser Frontend
       │
       ▼
FastAPI API
       │
       ▼
Services
 ┌───────────────┐
 │ Ingestion     │
 │ RAG           │
 └───────────────┘
       │
       ▼
Core Components
 ┌──────────────────────┐
 │ PDF Reader           │
 │ Chunker              │
 │ Embeddings           │
 │ Vector Store         │
 │ Retriever            │
 │ Prompt Builder       │
 │ Generator            │
 └──────────────────────┘
~~~

### API Layer

Responsible for:

- HTTP endpoints
- Request validation
- File uploads
- Response serialization
- HTTP error handling

### Service Layer

Responsible for application workflows.

**Ingestion Service**

~~~text
PDF
 ↓
Extract pages
 ↓
Create chunks
 ↓
Generate embeddings
 ↓
Store in ChromaDB
~~~

**RAG Service**

~~~text
Question
 ↓
Generate query embedding
 ↓
Retrieve candidate chunks
 ↓
Apply distance filtering
 ↓
Build grounded prompt
 ↓
Generate answer
 ↓
Return answer + citations + retrieved chunks
~~~

### Core Layer

The core modules contain the individual RAG components independently of HTTP.

This makes the system easier to test, replace, and extend.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ingest` | Upload and ingest a PDF |
| POST | `/api/query` | Ask a question and retrieve a grounded answer |
| GET | `/api/stats` | View vector store statistics and configuration |
| GET | `/api/health` | Check application and vector store health |
| DELETE | `/api/reset` | Clear the current vector collection |

Interactive API documentation is available through FastAPI Swagger:

~~~text
http://localhost:8000/docs
~~~

---

## Evaluation

The project includes a small evaluation framework for measuring retrieval quality.

Run the retrieval evaluation with:

~~~bash
python -m eval.run_eval
~~~

Run the optional end-to-end answer evaluation with:

~~~bash
python -m eval.run_eval --with-answers
~~~

Evaluate using a different Top-K value:

~~~bash
python -m eval.run_eval --top-k 5
~~~

### Metrics

**Hit Rate**

Measures whether the expected source content was retrieved within the Top-K results.

**Mean Reciprocal Rank (MRR)**

Measures how highly the expected source content appeared in the retrieved results.

The evaluation intentionally separates retrieval quality from generation quality.

The golden evaluation dataset is stored in:

~~~text
eval/golden_set.json
~~~

---

## Design Decisions

### Page-Aware Chunking

Chunks retain their source page number so that generated answers can provide useful document citations.

### Top-K + Distance Filtering

Top-K retrieval provides candidate context while the distance threshold prevents weakly related chunks from being passed to the generator.

### Retrieved Context Is Visible

The frontend displays the retrieved chunks instead of hiding them.

This makes it possible to inspect why an answer was generated and debug retrieval quality.

### Re-ingestion Replaces the Index

Uploading a new handbook resets the current vector collection and rebuilds it from the uploaded document.

This keeps the application behaviour predictable for the current single-document use case.

### No LangChain

The project intentionally implements the RAG pipeline directly.

This keeps the individual stages explicit:

- document processing
- chunking
- embedding
- retrieval
- prompt construction
- generation
- evaluation

The goal is to make the underlying RAG architecture easy to understand and modify.

---

## Testing

Run the test suite with:

~~~bash
pytest
~~~

The tests currently cover core functionality such as:

- text chunking
- retrieval evaluation metrics

The architecture allows additional tests to be added independently for:

- PDF extraction
- embeddings
- retrieval
- prompt construction
- API endpoints
- end-to-end RAG behaviour

---

## Security

API credentials are loaded from environment variables rather than hard-coded into the application.

The following files and directories are excluded from Git:

~~~text
.env
.venv/
data/chroma_db/
data/uploads/
__pycache__/
~~~

Use `.env.example` as the template for local configuration.

**Never commit a real OpenAI API key to the repository.**

---

## Future Improvements

Potential extensions include:

- Streaming responses
- Multiple document collections
- Document management UI
- Hybrid keyword + semantic retrieval
- Reranking
- Larger evaluation datasets
- Automated CI evaluation
- Conversation history
- Additional document formats
- PostgreSQL / pgvector
- Containerized deployment
- Authentication and role-based access control
- Observability and request tracing

---

## Project Goal

This project demonstrates how a simple RAG prototype can be evolved into a modular application with:

- explicit architecture
- configurable retrieval
- grounded generation
- source citations
- evaluation metrics
- automated tests
- API documentation
- frontend integration
- secure configuration

The emphasis is on understanding and controlling each stage of the RAG pipeline rather than hiding the implementation behind a framework abstraction.
