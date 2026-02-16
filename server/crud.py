from models import (
    LLMConfig,
    Message,
    Messages,
    Session,
    SourceCitation,
    UploadedDocument,
)


# --- Session CRUD ---


async def create_session(name: str = "New Chat") -> Session:
    session = Session(name=name)
    return await session.create()


async def list_sessions() -> list[Session]:
    return await Session.find().sort(-Session.created_at).to_list()


async def delete_session(session_id: str):
    session = await Session.find_one(Session.session_id == session_id)
    if session:
        # Cascade: collect session docs for FAISS + disk cleanup
        docs = await UploadedDocument.find(
            UploadedDocument.session_id == session_id
        ).to_list()
        session_docs = [(doc.chunk_ids, doc.filename) for doc in docs]
        await UploadedDocument.find(
            UploadedDocument.session_id == session_id
        ).delete()
        await Messages.find(Messages.session_id == session_id).delete()
        await session.delete()
        return session_docs
    return []


async def rename_session(session_id: str, name: str):
    session = await Session.find_one(Session.session_id == session_id)
    if session:
        session.name = name
        await session.save()
        return session


# --- Document CRUD ---


async def register_document(
    doc_id: str, filename: str, file_type: str, chunk_ids: list[str], session_id: str = ""
) -> UploadedDocument:
    doc = UploadedDocument(
        doc_id=doc_id, filename=filename, file_type=file_type, chunk_ids=chunk_ids, session_id=session_id
    )
    return await doc.create()


async def list_documents(session_id: str = "") -> list[UploadedDocument]:
    return await UploadedDocument.find(
        UploadedDocument.session_id == session_id
    ).sort(-UploadedDocument.uploaded_at).to_list()


async def delete_document(doc_id: str) -> UploadedDocument | None:
    doc = await UploadedDocument.find_one(UploadedDocument.doc_id == doc_id)
    if doc:
        await doc.delete()
        return doc
    return None


# --- Message CRUD ---


async def create_new_msg(
    message: Message,
    session_id: str = "",
    sources: list[SourceCitation] | None = None,
):
    last_msgs = (
        await Messages.find(Messages.session_id == session_id)
        .sort(-Messages.seqno)
        .limit(1)
        .to_list()
    )
    new_seqno = last_msgs[0].seqno + 1 if last_msgs else 0
    new_msg = Messages(
        seqno=new_seqno,
        username=message.username,
        message=message.message,
        session_id=session_id,
        sources=sources or [],
    )
    return await new_msg.create()


# --- LLM Config ---


async def get_llm_config() -> tuple[str, float]:
    cfg = await LLMConfig.find_one(LLMConfig.uid == 0)
    return cfg.model, cfg.temperature


async def set_llm_config(model: str, temperature: float):
    cfg = await LLMConfig.find_one(LLMConfig.uid == 0)
    cfg.model = model
    cfg.temperature = temperature
    await cfg.save()


async def init_llm_config():
    cfg = await LLMConfig.find_one(LLMConfig.uid == 0)
    if not cfg:
        cfg = LLMConfig(uid=0, model="claude-sonnet-4-5-20250929", temperature=0.2)
        await cfg.create()
