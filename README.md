# DocuVerse

Chat with your documents (PDF, DOCX, TXT, Markdown) using Claude and LangChain. Built with FastAPI, Streamlit, and FAISS.

## Features

- **Multi-format support** — Upload PDF, DOCX, TXT, and Markdown files
- **Chat sessions** — Create and switch between independent chat sessions
- **SSE streaming** — Real-time streaming responses via Server-Sent Events
- **Source citations** — See which documents and pages were used to answer each question
- **Document management** — Upload, list, and delete documents with FAISS sync
- **Summarize** — Get AI summaries of individual documents
- **Compare** — Compare and contrast multiple selected documents
- **Mind map** — Generate interactive mind maps from document content
- **Podcast** — Generate a two-host podcast from any document (via ElevenLabs)
- **Context filtering** — Select specific documents to scope your questions
- **Export** — Download chat history as Markdown
- **Configurable LLM** — Switch between Claude models and adjust temperature

## Tech Stack

- **Backend** — FastAPI + Uvicorn
- **Frontend** — Streamlit
- **LLM** — Claude (via LangChain + Anthropic)
- **Vector Store** — FAISS
- **Database** — MongoDB (via Beanie ODM)
- **Embeddings** — HuggingFace sentence-transformers
- **Streaming** — SSE (sse-starlette + sseclient-py)
- **Voice** — ElevenLabs (for podcast generation)
- **Package Manager** — [uv](https://docs.astral.sh/uv/)

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- MongoDB server running locally (or remote)
- Anthropic API key
- ElevenLabs API key (optional, for podcast feature)

## Setup

```sh
uv sync
cp .env.template .env
```

Edit `.env` and fill in your values:

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | (required) |
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `FILES_STORAGE_DIR` | Path to store uploaded files | `../uploaded_files` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (for podcast) | (optional) |

## Running

Open two terminals from the project root.

**Terminal 1 — Backend:**

```sh
cd server && uv run uvicorn main:app --reload
```

**Terminal 2 — Frontend:**

```sh
cd client && uv run streamlit run app.py
```

Or use the run script to start both at once:

```sh
./run.sh
```

- Backend: http://localhost:8000 (Swagger docs at http://localhost:8000/docs)
- Frontend: http://localhost:8501

## How It Works

1. Upload documents (PDF, DOCX, TXT, MD) through the sidebar
2. Files are chunked, embedded, and stored in FAISS; metadata is saved in MongoDB
3. Create a chat session and ask questions
4. LangChain retrieves relevant chunks and streams Claude's answer back via SSE
5. Source citations show which documents and pages informed each answer

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/sessions/` | List sessions |
| POST | `/sessions/` | Create session |
| DELETE | `/sessions/{session_id}` | Delete session + messages |
| PATCH | `/sessions/{session_id}` | Rename session |
| GET | `/sessions/{session_id}/messages/{offset}` | Get messages |
| POST | `/sessions/{session_id}/messages/` | Post message |
| GET | `/sessions/{session_id}/stream` | SSE streaming response |
| GET | `/sessions/{session_id}/export` | Export chat as markdown |
| GET | `/documents/` | List uploaded documents |
| DELETE | `/documents/{doc_id}` | Delete document |
| POST | `/documents/{doc_id}/summarize` | Summarize a document |
| POST | `/documents/{doc_id}/mindmap` | Generate mind map |
| POST | `/documents/{doc_id}/podcast` | Generate podcast |
| POST | `/documents/compare` | Compare selected documents |
| POST | `/process_documents/` | Process uploaded files |
| POST | `/set_llm/` | Update LLM settings |

## Project Structure

```
├── server/
│   ├── main.py          # FastAPI app with lifespan and routes
│   ├── config.py        # Environment configuration
│   ├── models.py        # Pydantic V2 & Beanie models
│   ├── crud.py          # Database operations (sessions, documents, messages)
│   ├── lang.py          # LangChain RAG pipeline, FAISS, multi-format loaders
│   └── tasks.py         # Response generation, streaming, summarize, compare, mindmap, podcast
├── client/
│   ├── app.py           # Streamlit frontend
│   └── .streamlit/      # Streamlit config (light theme default)
├── run.sh               # Start both backend and frontend
├── pyproject.toml       # Project config & dependencies (uv)
└── .env.template        # Environment variable template
```
