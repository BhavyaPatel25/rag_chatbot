# Hybrid RAG Chatbot (Session Memory)

A small hybrid Retrieval-Augmented Generation (RAG) chatbot API that combines a document retriever (Chroma + OpenAI embeddings) with an LLM interface (Ollama client). The service supports per-session ephemeral memory (last 10 messages) via cookies and returns context-aware answers using the contents of a resume document stored in `data/`.

This repository is intended as a lightweight example / starting point for building a RAG-based assistant that keeps short session memory.

---

## Highlights

- FastAPI-based HTTP API with two endpoints:
  - `GET /` — health check
  - `POST /chat` — ask a question (session cookie-based memory)
- Document retrieval using Chroma vector store and OpenAI embeddings
- LLM calls through the Ollama API client (configurable model)
- Per-session in-memory conversation memory (FIFO, last 10 messages)
- Vectorstore persisted to `vectorstore/` (created automatically on first run)

---

## Repository structure

- `app.py` — FastAPI application (handles session cookie creation and /chat endpoint)
- `retriever.py` — document loader, chunking, embeddings, and Chroma vectordb setup
- `ollama_llm.py` — LLM client wrapper and prompt construction
- `rag_pipeline.py` — ties retrieval, memory, and LLM together (main business logic)
- `memory.py` — in-memory per-session message store (deque with maxlen=10)
- `data/` — place your source documents here (by default expects `Bhavya Patel Resume.docx`)
- `vectorstore/` — persisted Chroma vectorstore (ignored in git)
- `requirement.txt` — Python dependencies
- `.gitignore` — ignores `.env`, `.venv`, `vectorstore/`, `__pycache__/`

---

## Requirements

- Python 3.10+
- An OpenAI API key for embeddings (used by `OpenAIEmbeddings`)
- An Ollama API key (used by the `ollama` client) and access to the configured model
- Required Python packages listed in `requirement.txt`

Example `requirement.txt` content:
- fastapi
- uvicorn
- ollama
- langchain
- langchain-community
- langchain-core
- langchain-chroma
- langchain-openai
- chromadb
- docx2txt
- python-dotenv
- tiktoken

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirement.txt
```

---

## Environment variables

Create a `.env` file in the project root (this file is gitignored). Minimum variables:

```text
OPENAI_API_KEY=sk-...
OLLAMA_API_KEY=...
OLLAMA_MODEL=gpt-oss:120b      # optional, defaults to "gpt-oss:120b"
```

- `OPENAI_API_KEY` — used by the `OpenAIEmbeddings` class to compute embeddings for document chunks.
- `OLLAMA_API_KEY` — used by the Ollama client wrapper to call the chosen LLM.
- `OLLAMA_MODEL` — optional model identifier for Ollama calls.

---

## Data

Place the resume (or other doc) you want to use as the retriever source at:

`data/Bhavya Patel Resume.docx`

On first run (when `vectorstore/` doesn't exist) the repo will:
1. Load `data/Bhavya Patel Resume.docx` via `docx2txt`.
2. Split the document into chunks (chunk_size=500, overlap=50).
3. Create a Chroma vector store under `vectorstore/` and persist it.

Note: `vectorstore/` is in `.gitignore` — do not commit that directory.

---

## How it works (high level)

1. Client calls POST `/chat` with JSON: `{"question": "<your question>"}`.
2. The server checks for a `session_id` cookie:
   - If missing, a new UUID session ID cookie is set on the response (httponly, secure, samesite=none).
3. The RAG pipeline:
   - Loads session memory (last 10 messages) from `memory.py` (in-memory only).
   - Uses `retriever.retrieve_context(question)` to get top-k relevant chunks from Chroma (k=3).
   - Calls the LLM wrapper (`ollama_llm.generate_answer`) with system prompt + context + memory + current user question.
   - Appends user question and assistant answer to session memory (`append_memory`).
4. API returns JSON: `{"answer": "<assistant answer>"}`

Key implementation notes:
- Retriever uses OpenAI embeddings and Chroma.
- Memory is ephemeral and per-process; it will not survive server restarts or multiple worker processes.
- The system prompt in `ollama_llm.py` includes instructions to only answer from provided context (and to say if something is not known).

---

## Running the server (development)

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

By default CORS is configured to allow:
- `https://bhavyapatel25.netlify.app` (modify `app.add_middleware` in `app.py` if you need to allow other origins)

---

## API usage

Health check:
```bash
curl http://localhost:8000/
# {"status":"Hybrid RAG API running (auto session enabled)"}
```

Chat endpoint (example using curl + cookie jar):

1) Ask a question (store cookies):
```bash
curl -c cookies.txt -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Bhavya Patel's experience?"}'
```

The server sets a `session_id` cookie in the response. To continue the same session in follow-up requests, include that cookie:

2) Follow-up question (re-using the same session):
```bash
curl -b cookies.txt -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"Can you summarize their education?"}'
```

Response JSON:
```json
{
  "answer": "..."
}
```

---

- Created a README describing the project, setup, how it works, and how to run it based on the repository files.

If you want, I can:
- Add a quick start script, an example `.env.example`, or update code to use a persistent memory backend (Redis) — tell me which and I will prepare it.
