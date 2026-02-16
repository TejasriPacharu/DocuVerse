import base64
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
import sseclient
import streamlit as st
from dotenv import load_dotenv
from streamlit_markmap import markmap

load_dotenv()

STORAGE_DIR = Path(os.environ.get("FILES_STORAGE_DIR", "../uploaded_files/"))
os.makedirs(STORAGE_DIR, exist_ok=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/")

st.set_page_config(
    page_title="LLM PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

def api(method, path, **kwargs):
    url = API_BASE_URL + path
    return getattr(requests, method)(url, **kwargs)


def init_session_state():
    defaults = {
        "messages": [],
        "offset": -1,
        "current_session": None,
        "sessions": [],
        "documents": [],
        "_last_sources": [],
        "_summarize_doc": None,
        "_summarize_name": None,
        "_mindmap_doc": None,
        "_mindmap_name": None,
        "_podcast_doc": None,
        "_podcast_name": None,
        "_compare_docs": None,
        "_session_select": None,
        "_results": [],  # persisted action results (summary, mindmap, audio, compare)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_sessions():
    try:
        resp = api("get", "sessions/")
        st.session_state.sessions = resp.json()
    except Exception:
        st.session_state.sessions = []


def load_documents():
    session_id = st.session_state.current_session
    if not session_id:
        st.session_state.documents = []
        return
    try:
        resp = api("get", f"sessions/{session_id}/documents/")
        st.session_state.documents = resp.json()
    except Exception:
        st.session_state.documents = []


def switch_session(session_id: str):
    st.session_state.current_session = session_id
    st.session_state.messages = []
    st.session_state.offset = -1
    st.session_state._results = []


def fetch_messages(session_id: str):
    try:
        resp = api("get", f"sessions/{session_id}/messages/{st.session_state.offset}")
        msgs = resp.json()
        for msg in msgs:
            st.session_state.messages.append(msg)
        if msgs:
            st.session_state.offset = msgs[-1]["seqno"]
    except Exception:
        pass


def sse_stream_generator(session_id: str, prompt: str):
    params = {"prompt": prompt}
    url = API_BASE_URL + f"sessions/{session_id}/stream?" + urlencode(params)
    response = requests.get(url, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        data = json.loads(event.data)
        if "token" in data:
            yield data["token"]
        if data.get("done"):
            st.session_state._last_sources = data.get("sources", [])
            break


def create_sidebar():
    with st.sidebar:
        # --- Session Manager ---
        st.markdown("## Sessions")

        load_sessions()
        sessions = st.session_state.sessions

        session_ids = [s["session_id"] for s in sessions]
        session_names_map = {s["session_id"]: s["name"] for s in sessions}

        if sessions:
            # Sync widget key with current_session before render
            if (
                st.session_state.current_session in session_ids
                and st.session_state.get("_session_select") != st.session_state.current_session
            ):
                st.session_state._session_select = st.session_state.current_session

            # Default to first session if current_session is invalid
            if st.session_state.current_session not in session_ids:
                st.session_state._session_select = session_ids[0]
                st.session_state.current_session = session_ids[0]

            col_select, col_del = st.columns([5, 1])
            with col_select:
                selected_id = st.selectbox(
                    "Session",
                    options=session_ids,
                    format_func=lambda sid: session_names_map.get(sid, sid),
                    key="_session_select",
                    label_visibility="collapsed",
                )
            with col_del:
                if st.button("X", key="del_session", help="Delete this session"):
                    api("delete", f"sessions/{st.session_state.current_session}")
                    st.session_state.current_session = None
                    st.session_state._session_select = None
                    st.session_state.messages = []
                    st.session_state.offset = -1
                    st.session_state._results = []
                    st.rerun()

            if selected_id != st.session_state.current_session:
                switch_session(selected_id)
                st.rerun()

        if st.button("New Chat", use_container_width=True):
            short_id = uuid.uuid4().hex[:4]
            resp = api("post", "sessions/", params={"name": f"Untitled Chat {short_id}"})
            new_session = resp.json()
            switch_session(new_session["session_id"])
            st.session_state._session_select = new_session["session_id"]
            st.rerun()

        st.divider()

        # --- Document Manager ---
        st.markdown("## Documents")

        files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md"],
            label_visibility="collapsed",
        )

        if st.button("Upload & Process", use_container_width=True, type="primary"):
            if files and st.session_state.current_session:
                with st.spinner("Processing files..."):
                    for file in files:
                        save_path = STORAGE_DIR / file.name
                        with open(save_path, mode="wb") as wf:
                            wf.write(file.getvalue())
                    resp = api(
                        "post",
                        "process_documents/",
                        json={"session_id": st.session_state.current_session},
                    )
                    result = resp.json()
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.toast("Files processed!")
                    st.rerun()
            elif not st.session_state.current_session:
                st.warning("Please create or select a session first")
            else:
                st.warning("Please select files first")

        load_documents()
        docs = st.session_state.documents

        if docs:
            for doc in docs:
                st.caption(f"{doc['filename']}")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    if st.button("X", key=f"del_{doc['doc_id']}", help="Delete"):
                        api("delete", f"documents/{doc['doc_id']}")
                        st.rerun()
                with c2:
                    if st.button("S", key=f"sum_{doc['doc_id']}", help="Summarize"):
                        st.session_state._summarize_doc = doc["doc_id"]
                        st.session_state._summarize_name = doc["filename"]
                with c3:
                    if st.button("M", key=f"map_{doc['doc_id']}", help="Mind Map"):
                        st.session_state._mindmap_doc = doc["doc_id"]
                        st.session_state._mindmap_name = doc["filename"]
                with c4:
                    if st.button("A", key=f"audio_{doc['doc_id']}", help="Audio Overview"):
                        st.session_state._podcast_doc = doc["doc_id"]
                        st.session_state._podcast_name = doc["filename"]
        else:
            st.info("No documents uploaded")

        st.divider()

        # --- LLM Settings ---
        st.markdown("## LLM Settings")

        model = st.selectbox(
            "Model",
            options=[
                "claude-sonnet-4-5-20250929",
                "claude-haiku-4-5-20251001",
                "claude-opus-4-6",
            ],
        )
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.2
        )

        if st.button("Save Settings", use_container_width=True):
            resp = api("post", "set_llm/", params={"temperature": temperature, "model": model})
            result = resp.json()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Settings saved!")



def render_sources(sources):
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            filename = s.get("filename", s) if isinstance(s, dict) else s.filename
            page = s.get("page") if isinstance(s, dict) else s.page
            snippet = s.get("snippet", "") if isinstance(s, dict) else s.snippet
            page_info = f" (p. {page})" if page is not None else ""
            st.markdown(f"**{filename}{page_info}**")
            st.caption(snippet[:150])


def run_chat():
    session_id = st.session_state.current_session
    if not session_id:
        st.info("Create or select a session to start chatting.")
        return

    fetch_messages(session_id)

    for msg in st.session_state.messages:
        role = "assistant" if msg["username"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["message"])
            if role == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"username": "user", "message": prompt, "sources": []})

        # Save user message to server (stream=True skips background task)
        api(
            "post",
            f"sessions/{session_id}/messages/",
            json={"username": "user", "message": prompt, "stream": True},
        )

        with st.chat_message("assistant"):
            try:
                st.session_state._last_sources = []
                response_text = st.write_stream(
                    sse_stream_generator(session_id, prompt)
                )
                sources = st.session_state.get("_last_sources", [])
                render_sources(sources)
                st.session_state.messages.append({
                    "username": "assistant",
                    "message": response_text,
                    "sources": sources,
                })
                # Update offset to skip re-fetching these messages
                st.session_state.offset += 2
            except Exception as e:
                st.error(f"Streaming error: {e}")


def _add_result(result_entry: dict):
    st.session_state._results.append(result_entry)


def handle_summarize():
    doc_id = st.session_state.get("_summarize_doc")
    if not doc_id:
        return
    doc_name = st.session_state.get("_summarize_name", "Document")
    with st.spinner(f"Summarizing {doc_name}..."):
        resp = api("post", f"documents/{doc_id}/summarize")
        result = resp.json()
    if "error" in result:
        st.error(result["error"])
    else:
        _add_result({"type": "summary", "name": doc_name, "content": result["summary"]})
    st.session_state._summarize_doc = None


def handle_compare():
    doc_ids = st.session_state.get("_compare_docs")
    if not doc_ids:
        return
    with st.spinner("Comparing documents..."):
        resp = api("post", "documents/compare", json={"doc_ids": doc_ids})
        result = resp.json()
    if "error" in result:
        st.error(result["error"])
    else:
        _add_result({"type": "comparison", "name": "Document Comparison", "content": result["comparison"]})
    st.session_state._compare_docs = None


def handle_mindmap():
    doc_id = st.session_state.get("_mindmap_doc")
    if not doc_id:
        return
    doc_name = st.session_state.get("_mindmap_name", "Document")
    with st.spinner(f"Generating mind map for {doc_name}..."):
        resp = api("post", f"documents/{doc_id}/mindmap")
        result = resp.json()
    if "error" in result:
        st.error(result["error"])
    else:
        _add_result({"type": "mindmap", "name": doc_name, "content": result["mindmap"]})
    st.session_state._mindmap_doc = None


def handle_audio_overview():
    doc_id = st.session_state.get("_podcast_doc")
    if not doc_id:
        return
    doc_name = st.session_state.get("_podcast_name", "Document")
    with st.spinner(f"Generating audio overview for {doc_name}..."):
        resp = api("post", f"documents/{doc_id}/audio-overview")
        result = resp.json()
    if "error" in result:
        st.error(result["error"])
    else:
        _add_result({
            "type": "audio_overview",
            "name": doc_name,
            "audio_base64": result["audio_base64"],
            "script": result["script"],
        })
    st.session_state._podcast_doc = None


def render_results():
    results = st.session_state.get("_results", [])
    if not results:
        return
    for i, entry in enumerate(results):
        with st.container(border=True):
            col_title, col_close = st.columns([9, 1])
            rtype = entry["type"]
            name = entry["name"]
            with col_title:
                if rtype == "summary":
                    st.markdown(f"### Summary: {name}")
                elif rtype == "comparison":
                    st.markdown(f"### {name}")
                elif rtype == "mindmap":
                    st.markdown(f"### Mind Map: {name}")
                elif rtype == "audio_overview":
                    st.markdown(f"### Audio Overview: {name}")
            with col_close:
                if st.button("X", key=f"close_result_{i}", help="Dismiss"):
                    st.session_state._results.pop(i)
                    st.rerun()

            if rtype in ("summary", "comparison"):
                st.markdown(entry["content"])
            elif rtype == "mindmap":
                markmap(entry["content"], height=500)
            elif rtype == "audio_overview":
                audio_bytes = base64.b64decode(entry["audio_base64"])
                st.audio(audio_bytes, format="audio/mp3")
                with st.expander("Overview Script"):
                    st.markdown(entry["script"])


def main():
    init_session_state()
    create_sidebar()

    # Header
    col_title, col_export = st.columns([4, 1])
    with col_title:
        st.markdown("# LLM Document Chat")
        st.caption("Upload documents and ask questions — powered by Claude & LangChain")
    with col_export:
        if st.session_state.current_session and st.session_state.messages:
            try:
                resp = api("get", f"sessions/{st.session_state.current_session}/export")
                st.download_button(
                    "Export Chat",
                    data=resp.text,
                    file_name="chat_export.md",
                    mime="text/markdown",
                )
            except Exception:
                pass

    # Handle special actions (generate and store results)
    handle_summarize()
    handle_compare()
    handle_mindmap()
    handle_audio_overview()

    # Render persisted results
    render_results()

    # Main chat
    run_chat()


if __name__ == "__main__":
    main()
