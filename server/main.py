import base64
import os
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from beanie import init_beanie
from config import CONFIG
from crud import (
    create_new_msg,
    create_session,
    delete_document,
    delete_session,
    init_llm_config,
    list_documents,
    list_sessions,
    rename_session,
    set_llm_config,
)
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from lang import delete_document_from_vectordb, process_docs
from models import LLMConfig, Message, Messages, Session, UploadedDocument
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from tasks import (
    auto_generate_title,
    compare_documents,
    generate_audio_overview,
    generate_mindmap,
    generate_response,
    stream_response,
    summarize_document,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncIOMotorClient(CONFIG.mongo_uri)
    db = client[CONFIG.db_name]
    await init_beanie(
        database=db,
        document_models=[Messages, LLMConfig, Session, UploadedDocument],
    )
    await init_llm_config()
    await process_docs()
    yield
    client.close()


app = FastAPI(
    title="LLM PDF Chat API",
    version="0.2.0",
    description="Chat with your documents using LLMs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Session endpoints ---


@app.get("/sessions/")
async def get_sessions() -> list[Session]:
    return await list_sessions()


@app.post("/sessions/")
async def create_session_endpoint(name: str = "New Chat") -> Session:
    return await create_session(name)


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    session_docs = await delete_session(session_id)
    for chunk_ids, filename in session_docs:
        delete_document_from_vectordb(chunk_ids)
        filepath = os.path.join(CONFIG.files_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    return {"message": "deleted"}


@app.patch("/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, name: str):
    session = await rename_session(session_id, name)
    if session:
        return session
    return {"error": "session not found"}


# --- Message endpoints ---


@app.get("/sessions/{session_id}/messages/{offset}")
async def get_messages(session_id: str, offset: int = 0) -> list[Messages]:
    return await Messages.find(
        Messages.session_id == session_id,
        Messages.seqno > offset,
    ).to_list()


class PostMessageBody(BaseModel):
    username: str
    message: str
    stream: bool = False


@app.post("/sessions/{session_id}/messages/")
async def post_message(
    session_id: str, body: PostMessageBody, bg: BackgroundTasks
) -> Messages:
    msg = await create_new_msg(
        Message(username=body.username, message=body.message),
        session_id=session_id,
    )
    # Auto-title on first user message (seqno 0)
    if msg.seqno == 0 and body.username != "assistant":
        bg.add_task(auto_generate_title, session_id, body.message)
    # Resolve doc_ids from session's documents
    session_docs = await list_documents(session_id)
    doc_ids = [d.doc_id for d in session_docs] if session_docs else None
    if not body.stream:
        bg.add_task(
            generate_response,
            prompt=body.message,
            session_id=session_id,
            doc_ids=doc_ids,
        )
    return msg


# --- SSE Streaming ---


@app.get("/sessions/{session_id}/stream")
async def stream_endpoint(
    session_id: str,
    prompt: str,
):
    session_docs = await list_documents(session_id)
    doc_ids = [d.doc_id for d in session_docs] if session_docs else None
    return EventSourceResponse(
        stream_response(
            prompt=prompt,
            session_id=session_id,
            doc_ids=doc_ids,
        )
    )


# --- Export ---


@app.get("/sessions/{session_id}/export")
async def export_chat(session_id: str):
    messages = await Messages.find(
        Messages.session_id == session_id
    ).sort(Messages.seqno).to_list()

    lines = ["# Chat Export\n"]
    for msg in messages:
        role = "Assistant" if msg.username == "assistant" else "User"
        lines.append(f"## {role}\n\n{msg.message}\n")
        if msg.sources:
            lines.append("**Sources:**\n")
            for s in msg.sources:
                page_info = f" (page {s.page})" if s.page is not None else ""
                lines.append(f"- {s.filename}{page_info}: {s.snippet[:100]}...\n")
        lines.append("")

    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=chat_export.md"},
    )


# --- Document endpoints ---


@app.get("/sessions/{session_id}/documents/")
async def get_documents(session_id: str) -> list[UploadedDocument]:
    return await list_documents(session_id)


@app.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    doc = await delete_document(doc_id)
    if doc:
        delete_document_from_vectordb(doc.chunk_ids)
        filepath = os.path.join(CONFIG.files_dir, doc.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return {"message": "deleted"}
    return {"error": "document not found"}


@app.post("/documents/{doc_id}/summarize")
async def summarize_endpoint(doc_id: str):
    return await summarize_document(doc_id)


class CompareBody(BaseModel):
    doc_ids: list[str]


@app.post("/documents/compare")
async def compare_endpoint(body: CompareBody):
    return await compare_documents(body.doc_ids)


@app.post("/documents/{doc_id}/mindmap")
async def mindmap_endpoint(doc_id: str):
    return await generate_mindmap(doc_id)


@app.post("/documents/{doc_id}/audio-overview")
async def audio_overview_endpoint(doc_id: str):
    result = await generate_audio_overview(doc_id)
    if "error" in result:
        return result
    return {
        "audio_base64": base64.b64encode(result["audio"]).decode(),
        "script": result["script"],
    }


# --- Process documents ---


class ProcessDocumentsBody(BaseModel):
    session_id: str = ""


@app.post("/process_documents/")
async def process_documents_endpoint(body: ProcessDocumentsBody | None = None):
    try:
        session_id = body.session_id if body else ""
        await process_docs(session_id=session_id)
        return {"message": "success"}
    except Exception as err:
        return {"error": str(err)}


# --- LLM config ---


@app.post("/set_llm/")
async def set_llm(model: str, temperature: float):
    try:
        await set_llm_config(model, temperature)
        return {"message": "success"}
    except Exception as err:
        return {"error": str(err)}
