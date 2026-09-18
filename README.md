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
- Two-layer refusal behaviour when relevant context cannot be retrieved
- Retrieval evaluation using Hit Rate and MRR
- Top-K sweep to support a data-backed retrieval configuration
- Optional end-to-end answer and refusal evaluation
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

TOP_K=4
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
| `TOP_K` | Number of candidate chunks retrieved for a query. Set to 4 based on the evaluation results below |
| `MAX_DISTANCE` | Maximum vector distance allowed for retrieved chunks |
| `EMBEDDING_MODEL` | OpenAI model used for embeddings |
| `GENERATION_MODEL` | OpenAI model used for answer generation |
| `CHROMA_PATH` | Persistent ChromaDB storage location |
| `UPLOAD_DIR` | Location for uploaded PDFs |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires re-ingesting the document because existing embeddings were generated under the previous settings.

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

This makes the system easier to test, replace, and extend. Swapping ChromaDB for pgvector, or OpenAI for another provider, is a change to one module in `core/` rather than a change across the application.

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

The project includes an evaluation harness for measuring retrieval quality independently of generation quality.

### Running the Evaluation

Retrieval metrics only, with no LLM calls:

~~~bash
python -m eval.run_eval
~~~

End-to-end evaluation, adding answer and refusal checks:

~~~bash
python -m eval.run_eval --with-answers
~~~

Sweep Top-K to compare configurations:

~~~bash
for k in 1 2 3 4 5 10; do python -m eval.run_eval --top-k $k; done
~~~

Each run writes a timestamped JSON report to `eval/results/`, including the chunk size, overlap, Top-K, distance threshold, and models used, so that runs remain comparable after a configuration change.

### Metrics

**Hit Rate@K**

Whether the page containing the answer appeared anywhere in the Top-K retrieved chunks. This measures whether the relevant content reached the model at all.

**Mean Reciprocal Rank (MRR)**

The reciprocal of the rank at which the first correct chunk appeared, averaged across questions. This measures whether the relevant content appeared early in the retrieval results.

Retrieval is scored without invoking the LLM. When an answer is wrong, the first thing to establish is whether retrieval failed or generation failed. An end-to-end score alone can hide that distinction.

`--with-answers` adds two further checks: a keyword pass against the expected answer, and a refusal rate over deliberately out-of-scope questions.

A RAG system that confidently answers a question the document does not cover has a broken guardrail, and the refusal test is designed to detect that behaviour.

### Evaluation Results

**Evaluation dataset:** 8 questions from the employee handbook.

**Ground truth:** Expected pages were labelled by reading the source PDF directly.

**Evaluation configuration:**

- `CHUNK_SIZE=500`
- `CHUNK_OVERLAP=50`
- `MAX_DISTANCE=1.2`
- `TOP_K` evaluated at 1, 2, 3, 4, 5, and 10

| K | Hit Rate | MRR |
|---:|---:|---:|
| 1 | 0.750 | 0.750 |
| 2 | 0.750 | 0.750 |
| 3 | 0.875 | 0.792 |
| 4 | 1.000 | 0.823 |
| 5 | 1.000 | 0.823 |
| 10 | 1.000 | 0.823 |

### Top-K Selection

**`TOP_K=4` was selected for the current evaluation set.**

It achieves a Hit Rate of 1.0, while increasing K beyond 4 provides no additional recall across these eight questions.

The results also show that increasing K primarily improves coverage rather than ranking quality: MRR increases from 0.792 at K=3 to 0.823 at K=4, then remains unchanged.

This suggests that the additional relevant content introduced at K=4 is being retrieved later in the ranking rather than improving the quality of the highest-ranked result.

For a larger evaluation set, a reranking stage or improved chunking strategy could be investigated rather than simply increasing the retrieval window.

### Guardrail Behaviour

The application uses two independent layers of refusal behaviour:

| Test Question | Guardrail Layer | Behaviour |
|---|---|---|
| "Who won the 2019 cricket world cup?" | Distance threshold | No chunk cleared `MAX_DISTANCE`, so the request was refused without an LLM call |
| "What is the pet bereavement policy?" | Prompt grounding | Bereavement chunks were semantically close enough to pass the threshold, so the LLM was called. It found no pet policy in the retrieved context and stated that the information was not available |

The second test is particularly useful because retrieval was not necessarily incorrect: it returned the closest available content. The prompt-level grounding instruction then prevented the model from filling the information gap with an unsupported answer.

### Ground-Truth Labelling

An earlier evaluation run scored 0.75 Hit Rate with two apparent retrieval failures.

Investigation showed that both were ground-truth labelling errors rather than retrieval errors. Expected pages had originally been labelled using keyword search, which matched the table of contents and passing references rather than the pages containing the answers.

The evaluation set was subsequently re-labelled against the actual answer content.

The rule adopted since is:

> A page belongs in `expected_pages` only if someone reading that page alone could answer the question.

This makes the evaluation target the actual retrieval requirement rather than incidental keyword matches.

### Limitations of the Current Evaluation

- **n=8**, so a single question is worth 12.5 percentage points and the metrics move in large steps.
- All eight questions use the document's own vocabulary. A Hit Rate of 1.0 indicates that the current test set is relatively easy rather than proving that retrieval is solved.
- Answer correctness is checked using keyword matching, which is a weak proxy for faithfulness.
- `MAX_DISTANCE` was set by observation rather than systematically tuned against the golden set.
- The evaluation currently focuses on a single handbook.

Planned improvements include paraphrased questions, distractor questions, multi-section questions, expanding the evaluation set to 20 or more questions, and adding an LLM-as-judge faithfulness scorer.

The golden evaluation dataset is stored in:

~~~text
eval/golden_set.json
~~~

---

## Design Decisions

### Page-Aware Chunking

The original notebook joined all pages into a single string before chunking, which made page-level citation impossible.

Chunking within each page means every chunk carries its source page and the interface can cite it.

The trade-off is that a sentence spanning a page break can be split across two chunks. For a handbook of discrete policy sections, this is worth paying for citations. For a continuous narrative document, it would be less desirable.

### Top-K Plus Distance Filtering

Top-K retrieval always returns the nearest chunks, even when the nearest chunk is irrelevant.

Without a distance threshold, an off-topic question can still reach the model wrapped in confident-looking context.

`MAX_DISTANCE` discards chunks that are merely the least-bad matches, and the service refuses rather than generating when no chunk clears the threshold.

The `grounded` flag on the response records which retrieval path was taken.

`TOP_K` is set from the evaluation sweep above rather than chosen purely by default.

### Retrieved Context Is Visible

The frontend displays the retrieved chunks with their chunk ID, page number, and distance instead of hiding them.

This serves two purposes:

1. Users can verify an answer against its source.
2. When an answer is wrong, the retrieved chunks show whether the problem occurred during retrieval or generation.

### Re-ingestion Replaces the Index

Uploading a handbook resets the vector collection and rebuilds it.

Appending would create duplicate chunks from the same source document, which could cause duplicate content to compete for retrieval positions and crowd out other relevant chunks.

### Retrieval Evaluated Separately From Generation

Retrieval metrics are computed without calling the LLM, so failures can be attributed to the correct stage.

This makes it possible to distinguish:

- retrieval failure
- grounding/guardrail failure
- generation failure

rather than treating all incorrect answers as the same problem.

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

The goal is to make the underlying RAG architecture easy to understand, inspect, test, and modify.

---

## Testing

Run the test suite with:

~~~bash
pytest
~~~

The tests currently cover core functionality such as:

- text chunking boundaries and overlap validation
- page metadata propagation
- retrieval evaluation metrics

No network calls are made by the test suite.

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
eval/results/
~~~

Use `.env.example` as the template for local configuration.

**Never commit a real OpenAI API key to the repository.**

---

## Future Improvements

Potential extensions include:

- Reranking stage to improve MRR on weakly ranked retrievals
- Token-aware or heading-aware chunking
- LLM-as-judge faithfulness scoring
- Expanded golden set with paraphrase, distractor, and multi-section questions
- Automated evaluation in CI, gated on Hit Rate and MRR thresholds
- Hybrid keyword and semantic retrieval
- Streaming responses
- Multiple document collections and source filtering on queries
- Document management UI
- Conversation history
- Additional document formats
- PostgreSQL / pgvector
- Containerized deployment
- Authentication and role-based access control
- Observability and request tracing, including aggregated latency and cost per query

---

## Project Goal

This project demonstrates how a simple RAG prototype can be evolved into a modular application with:

- explicit architecture
- retrieval configured from measured results rather than defaults
- grounded generation with two independent guardrail layers
- source citations
- an evaluation harness separating retrieval from generation
- automated tests
- API documentation
- frontend integration
- secure configuration

The emphasis is on understanding and controlling each stage of the RAG pipeline rather than hiding the implementation behind a framework abstraction.
