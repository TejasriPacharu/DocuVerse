from datetime import datetime
from typing import Annotated
from uuid import uuid4

from beanie import Document, Indexed
from pydantic import BaseModel, Field


class Session(Document):
    session_id: Annotated[str, Indexed(unique=True)] = Field(
        default_factory=lambda: str(uuid4())
    )
    name: str = "New Chat"
    created_at: datetime = Field(default_factory=datetime.now)


class UploadedDocument(Document):
    doc_id: Annotated[str, Indexed(unique=True)] = Field(
        default_factory=lambda: str(uuid4())
    )
    session_id: Annotated[str, Indexed()] = ""
    filename: str
    file_type: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    chunk_ids: list[str] = []


class SourceCitation(BaseModel):
    filename: str
    page: int | None = None
    snippet: str


class Message(BaseModel):
    username: str
    message: str


class Messages(Document, Message):
    seqno: Annotated[int, Indexed()]
    session_id: Annotated[str, Indexed()] = ""
    sources: list[SourceCitation] = []


class LLMConfig(Document):
    uid: Annotated[int, Indexed()]
    model: str
    temperature: float
