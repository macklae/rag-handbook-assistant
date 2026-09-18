const $ = (id) => document.getElementById(id);

const pdfInput = $("pdfInput");
const uploadBtn = $("uploadBtn");
const resetBtn = $("resetBtn");
const ingestStatus = $("ingestStatus");
const statsEl = $("stats");
const questionEl = $("question");
const topKEl = $("topK");
const askBtn = $("askBtn");
const answerCard = $("answerCard");
const answerMeta = $("answerMeta");
const answerEl = $("answer");
const chunksEl = $("chunks");

function setStatus(message, kind) {
  ingestStatus.hidden = false;
  ingestStatus.className = `status ${kind}`;
  ingestStatus.textContent = message;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

async function call(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return body;
}

async function loadStats() {
  try {
    const stats = await call("/api/stats");
    const sources = stats.sources.length ? stats.sources.join(", ") : "none";
    statsEl.innerHTML =
      `<span><b>${stats.chunk_count}</b> chunks indexed</span>` +
      `<span>Source: <b>${escapeHtml(sources)}</b></span>` +
      `<span>Chunk size <b>${stats.chunk_size}</b> / overlap <b>${stats.chunk_overlap}</b></span>` +
      `<span>Embedding: <b>${escapeHtml(stats.embedding_model)}</b></span>` +
      `<span>Generation: <b>${escapeHtml(stats.generation_model)}</b></span>`;
    topKEl.value = stats.top_k;
  } catch (error) {
    statsEl.textContent = `Could not load stats: ${error.message}`;
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = pdfInput.files[0];
  if (!file) {
    setStatus("Choose a PDF first.", "err");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  uploadBtn.disabled = true;
  setStatus(`Reading, chunking and embedding ${file.name}. This can take a minute.`, "busy");

  try {
    const result = await call("/api/ingest", { method: "POST", body: form });
    setStatus(
      `Indexed ${result.filename}: ${result.pages} pages, ` +
        `${result.characters.toLocaleString()} characters, ${result.chunks} chunks.`,
      "ok"
    );
    await loadStats();
  } catch (error) {
    setStatus(error.message, "err");
  } finally {
    uploadBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  try {
    const result = await call("/api/reset", { method: "DELETE" });
    setStatus(`Cleared ${result.removed} chunks from the index.`, "ok");
    answerCard.hidden = true;
    await loadStats();
  } catch (error) {
    setStatus(error.message, "err");
  } finally {
    resetBtn.disabled = false;
  }
});

async function ask() {
  const question = questionEl.value.trim();
  if (question.length < 3) {
    setStatus("Type a question first.", "err");
    return;
  }

  askBtn.disabled = true;
  answerCard.hidden = false;
  answerMeta.textContent = "";
  answerEl.textContent = "Retrieving and generating...";
  chunksEl.innerHTML = "";

  try {
    const result = await call("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topKEl.value) || undefined,
      }),
    });

    const badge = result.grounded
      ? '<span class="badge grounded">grounded</span>'
      : '<span class="badge refused">refused, nothing relevant retrieved</span>';

    answerMeta.innerHTML =
      badge +
      `<span>${result.chunks.length} chunks used</span>` +
      `<span>${result.latency_ms} ms</span>` +
      `<span>${escapeHtml(result.generation_model)}</span>`;

    answerEl.textContent = result.answer;

    chunksEl.innerHTML = result.chunks
      .map(
        (chunk) =>
          `<div class="chunk">
             <div class="chunk-head">
               <span>${escapeHtml(chunk.chunk_id)}</span>
               <span>page ${chunk.page}</span>
               <span>distance ${chunk.distance}</span>
             </div>
             <div class="chunk-text">${escapeHtml(chunk.text)}</div>
           </div>`
      )
      .join("");
  } catch (error) {
    answerEl.textContent = "";
    answerMeta.innerHTML = '<span class="badge refused">error</span>';
    chunksEl.innerHTML = `<div class="status err">${escapeHtml(error.message)}</div>`;
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", ask);
questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") ask();
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionEl.value = chip.dataset.q;
    ask();
  });
});

loadStats();
