import json
import os
import traceback
from collections.abc import AsyncGenerator

import lang
from crud import create_new_msg, get_llm_config, rename_session
from elevenlabs.client import ElevenLabs
from models import Message, SourceCitation


async def auto_generate_title(session_id: str, message: str):
    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
        resp = await llm.ainvoke(
            f"Generate a concise 3-5 word title for a chat that starts with this message. "
            f"Reply with ONLY the title, no quotes or punctuation.\n\nMessage: {message}"
        )
        title = resp.content.strip().strip('"\'')
        if title:
            await rename_session(session_id, title)
    except Exception:
        pass  # non-critical, keep default title


def _extract_sources(llm_response: dict) -> list[SourceCitation]:
    sources = []
    seen = set()
    context_docs = llm_response.get("context", [])
    for doc in context_docs:
        meta = doc.metadata
        filename = meta.get("filename", meta.get("source", "unknown"))
        page = meta.get("page", None)
        snippet = doc.page_content[:200]
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append(SourceCitation(filename=filename, page=page, snippet=snippet))
    return sources


async def generate_response(prompt: str, session_id: str = "", doc_ids: list[str] | None = None):
    try:
        model, temp = await get_llm_config()
        chain = lang.get_chain(doc_ids=doc_ids, model=model, temp=temp)
        if chain is None:
            message = "No documents have been processed yet. Please upload and process documents first."
            sources = []
        else:
            llm_response = await chain.ainvoke({"input": prompt})
            message = llm_response["answer"]
            sources = _extract_sources(llm_response)
    except Exception as err:
        print("".join(traceback.format_exception(err)))
        message = f"Sorry... There was some error unfortunately.\n```text\n{type(err).__name__}\n{err!s}\n```"
        sources = []
    await create_new_msg(
        Message(username="assistant", message=message),
        session_id=session_id,
        sources=sources,
    )


async def stream_response(
    prompt: str, session_id: str = "", doc_ids: list[str] | None = None
) -> AsyncGenerator[str, None]:
    try:
        model, temp = await get_llm_config()
        chain = lang.get_chain(doc_ids=doc_ids, model=model, temp=temp)
        if chain is None:
            msg = "No documents have been processed yet. Please upload and process documents first."
            yield json.dumps({'token': msg})
            yield json.dumps({'done': True, 'sources': []})
            await create_new_msg(
                Message(username="assistant", message=msg),
                session_id=session_id,
            )
            return

        full_answer = ""
        context_docs = []

        async for chunk in chain.astream({"input": prompt}):
            if "answer" in chunk:
                token = chunk["answer"]
                full_answer += token
                yield json.dumps({'token': token})
            if "context" in chunk:
                context_docs = chunk["context"]

        sources = []
        seen = set()
        for doc in context_docs:
            meta = doc.metadata
            filename = meta.get("filename", meta.get("source", "unknown"))
            page = meta.get("page", None)
            snippet = doc.page_content[:200]
            key = (filename, page)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceCitation(filename=filename, page=page, snippet=snippet)
                )

        yield json.dumps({'done': True, 'sources': [s.model_dump() for s in sources]})

        await create_new_msg(
            Message(username="assistant", message=full_answer),
            session_id=session_id,
            sources=sources,
        )
    except Exception as err:
        print("".join(traceback.format_exception(err)))
        error_msg = f"Error: {type(err).__name__}: {err!s}"
        yield json.dumps({'token': error_msg})
        yield json.dumps({'done': True, 'sources': []})
        await create_new_msg(
            Message(username="assistant", message=error_msg),
            session_id=session_id,
        )


async def summarize_document(doc_id: str) -> dict:
    try:
        model, temp = await get_llm_config()
        retriever = lang.get_retriever(doc_ids=[doc_id])
        if retriever is None:
            return {"error": "No vectordb available"}
        docs = retriever.invoke("summarize this document")
        summarize_chain = lang.create_summarize_chain(model, temp)
        result = await summarize_chain.ainvoke({"context": docs})
        return {"summary": result}
    except Exception as err:
        return {"error": str(err)}


async def compare_documents(doc_ids: list[str]) -> dict:
    try:
        model, temp = await get_llm_config()
        retriever = lang.get_retriever(doc_ids=doc_ids)
        if retriever is None:
            return {"error": "No vectordb available"}
        chain = lang.create_comparison_chain(retriever, model, temp)
        result = await chain.ainvoke(
            {"input": "Compare and contrast these documents in detail."}
        )
        return {"comparison": result["answer"]}
    except Exception as err:
        return {"error": str(err)}


async def generate_mindmap(doc_id: str) -> dict:
    try:
        model, temp = await get_llm_config()
        retriever = lang.get_retriever(doc_ids=[doc_id])
        if retriever is None:
            return {"error": "No vectordb available"}
        docs = retriever.invoke("all topics and key points in this document")
        chain = lang.create_mindmap_chain(model, temp)
        result = await chain.ainvoke({"context": docs})
        return {"mindmap": result}
    except Exception as err:
        return {"error": str(err)}


def _tts_to_bytes(client: ElevenLabs, text: str, voice_id: str) -> bytes:
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(chunk for chunk in audio if isinstance(chunk, bytes))


async def generate_audio_overview(doc_id: str) -> dict:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY not set"}

    try:
        model, temp = await get_llm_config()
        retriever = lang.get_retriever(doc_ids=[doc_id])
        if retriever is None:
            return {"error": "No vectordb available"}

        docs = retriever.invoke("all content and key topics in this document")
        overview_chain = lang.create_audio_overview_chain(model, max(temp, 0.5))
        script = await overview_chain.ainvoke({"context": docs})

        if not script or not script.strip():
            return {"error": "Failed to generate overview script"}

        client = ElevenLabs(api_key=api_key)
        voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George - warm narrator

        audio_bytes = _tts_to_bytes(client, script.strip(), voice_id)

        return {"audio": audio_bytes, "script": script}
    except Exception as err:
        print("".join(traceback.format_exception(err)))
        return {"error": str(err)}
