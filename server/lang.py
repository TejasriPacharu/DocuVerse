import hashlib
import os

from config import CONFIG
from crud import get_llm_config, register_document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_INDEX_DIR = "../faiss-index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb: FAISS | None = None

# Track chunk_id -> doc_id mapping for filtering and deletion
_chunk_doc_map: dict[str, str] = {}


def _get_loader(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(filepath)
    elif ext == ".docx":
        return Docx2txtLoader(filepath)
    elif ext in (".txt", ".md"):
        return TextLoader(filepath)
    return None


def _generate_chunk_id(text: str, source: str, index: int) -> str:
    content = f"{source}:{index}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()


async def process_docs(session_id: str = ""):
    global vectordb

    files_dir = CONFIG.files_dir
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_texts = []

    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        if not os.path.isfile(filepath):
            continue
        loader = _get_loader(filepath)
        if not loader:
            continue

        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        ext = os.path.splitext(filename)[1].lower()
        doc_id = hashlib.sha256(filename.encode()).hexdigest()[:16]

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            cid = _generate_chunk_id(chunk.page_content, filename, i)
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = filename
            chunk.metadata["chunk_id"] = cid
            _chunk_doc_map[cid] = doc_id
            chunk_ids.append(cid)

        all_texts.extend(chunks)

        try:
            await register_document(
                doc_id=doc_id,
                filename=filename,
                file_type=ext.lstrip("."),
                chunk_ids=chunk_ids,
                session_id=session_id,
            )
        except Exception:
            pass  # already registered

    if all_texts:
        if vectordb is None:
            vectordb = FAISS.from_documents(documents=all_texts, embedding=embeddings)
        else:
            new_db = FAISS.from_documents(documents=all_texts, embedding=embeddings)
            vectordb.merge_from(new_db)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectordb.save_local(FAISS_INDEX_DIR)
    elif vectordb is None:
        if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
            vectordb = FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )


def _filter_by_doc_ids(doc_ids: list[str]):
    def _filter(metadata: dict) -> bool:
        return metadata.get("doc_id") in doc_ids
    return _filter


def get_retriever(doc_ids: list[str] | None = None):
    if vectordb is None:
        return None
    if doc_ids:
        return vectordb.as_retriever(
            search_kwargs={"k": 7, "filter": _filter_by_doc_ids(doc_ids)}
        )
    return vectordb.as_retriever(search_kwargs={"k": 7})


def get_chain(doc_ids: list[str] | None = None, model: str = "claude-sonnet-4-5-20250929", temp: float = 0.0):
    retriever = get_retriever(doc_ids)
    if retriever is None:
        return None
    return create_chain(retriever, model, temp)


def create_chain(retriever, model="claude-sonnet-4-5-20250929", temp=0.0):
    llm = ChatAnthropic(model=model, temperature=temp)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_summarize_chain(model="claude-sonnet-4-5-20250929", temp=0.0):
    llm = ChatAnthropic(model=model, temperature=temp)
    system_prompt = (
        "You are an expert document summarizer. "
        "Provide a comprehensive yet concise summary of the following document content. "
        "Highlight key points, main arguments, and important details.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Please summarize this document."),
    ])
    return create_stuff_documents_chain(llm, prompt)


def create_comparison_chain(retriever, model="claude-sonnet-4-5-20250929", temp=0.0):
    llm = ChatAnthropic(model=model, temperature=temp)
    system_prompt = (
        "You are an expert document analyst. "
        "Compare and contrast the following documents. "
        "For each point, cite which document the information comes from. "
        "Highlight similarities, differences, and unique aspects of each document.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_mindmap_chain(model="claude-sonnet-4-5-20250929", temp=0.0):
    llm = ChatAnthropic(model=model, temperature=temp)
    system_prompt = (
        "You are an expert at creating mind maps from document content. "
        "Given the following document content, generate a hierarchical mind map "
        "in Markdown format using headings (#, ##, ###) and bullet points (- ). "
        "The mind map should capture the key topics, subtopics, and important details. "
        "Use a clear hierarchy: # for the main topic, ## for major themes, "
        "### for subtopics, and - for leaf details. "
        "Output ONLY the markdown mind map, no other text.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate a mind map for this document."),
    ])
    return create_stuff_documents_chain(llm, prompt)


def create_audio_overview_chain(model="claude-sonnet-4-5-20250929", temp=0.5):
    llm = ChatAnthropic(model=model, temperature=temp)
    system_prompt = (
        "You are an expert narrator. Given the following document content, "
        "write a clear, engaging two-minute audio overview script. "
        "Use a single narrator voice. Cover the key points, main arguments, "
        "and important takeaways. Write in a natural speaking style — "
        "conversational but informative. Aim for roughly 300 words "
        "(about two minutes when read aloud). "
        "Output ONLY the narration text, no stage directions or labels.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Write a two-minute audio overview of this document."),
    ])
    return create_stuff_documents_chain(llm, prompt)


def delete_document_from_vectordb(chunk_ids: list[str]):
    global vectordb
    if vectordb and chunk_ids:
        try:
            vectordb.delete(chunk_ids)
            vectordb.save_local(FAISS_INDEX_DIR)
        except Exception:
            pass
    for cid in chunk_ids:
        _chunk_doc_map.pop(cid, None)


async def load_chain():
    model, temp = await get_llm_config()
    return get_chain(model=model, temp=temp)
