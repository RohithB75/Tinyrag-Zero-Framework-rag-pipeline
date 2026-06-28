from __future__ import annotations

from pathlib import Path

import streamlit as st
import requests

from src.rag_pipeline import TinyRAG


APP_TITLE = "TinyRAG Studio"
APP_TAGLINE = "A compact, polished RAG interface for exploring retrieval, chunking, and answers in one place."


def load_sample_text() -> str:
    sample_path = Path(__file__).resolve().parent / "data" / "sample_doc.txt"
    return sample_path.read_text(encoding="utf-8")


def init_state() -> None:
    defaults = {
        "rag": None,
        "messages": [],
        "document_name": None,
        "document_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_index(text: str, source_name: str, chunk_size: int, overlap: int) -> None:
    rag = TinyRAG()
    with st.spinner("Indexing document and creating embeddings..."):
        rag.ingest_text(
            text,
            source_label=source_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    st.session_state.rag = rag
    st.session_state.document_name = source_name
    st.session_state.document_stats = {
        "chunks": len(rag.store),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    st.session_state.messages = []


def render_metrics() -> None:
    stats = st.session_state.document_stats
    if not stats:
        return

    left, middle, right = st.columns(3)
    left.metric("Chunks indexed", stats["chunks"])
    middle.metric("Chunk size", stats["chunk_size"])
    right.metric("Overlap", stats["overlap"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"], st.session_state.document_name or "document")


def render_sources(sources: list[tuple[float, dict]], document_name: str) -> None:
    with st.expander("Retrieved chunks", expanded=False):
        st.caption("These are the passages the model used to answer your question.")
        for index, (score, record) in enumerate(sources, start=1):
            metadata = record.get("metadata", {})
            chunk_id = metadata.get("chunk_id", index - 1)
            source = metadata.get("source", document_name)
            preview = record["text"].strip().replace("\n", " ")
            if len(preview) > 260:
                preview = preview[:260].rstrip() + "..."
            st.markdown(
                f"""
                <div class="chunk-card">
                    <div class="chunk-card__header">
                        <span>Passage {index}</span>
                        <span>Similarity {score:.3f}</span>
                    </div>
                    <div class="chunk-card__meta">Source: {source} · Chunk ID: {chunk_id}</div>
                    <div class="chunk-card__body">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_answer(question: str, top_k: int) -> tuple[str, list[tuple[float, dict]] | None, str | None]:
    try:
        answer, sources = st.session_state.rag.query(question, top_k=top_k, show_chunks=False)
        return answer, sources, None
    except requests.exceptions.RequestException as exc:
        return (
            "I could not reach the local Ollama server. Start Ollama, confirm the models are pulled, and try again.",
            None,
            str(exc),
        )
    except Exception as exc:
        return (
            "Something went wrong while generating the answer. Try rebuilding the index or retrying the question.",
            None,
            str(exc),
        )


def ask_and_store(question: str, top_k: int) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    answer, sources, error_detail = get_answer(question, top_k)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources or []})
    if error_detail:
        st.session_state.messages.append({"role": "assistant", "content": f"Debug info: {error_detail}", "sources": []})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="RAG")
    init_state()

    st.markdown(
        """
        <style>
            .stApp {
                background: #f4f7fb;
                color: #0f172a;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }

            [data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }

            [data-testid="stSidebar"] * {
                color: #0f172a !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                width: 100%;
                background: #0f172a;
                color: #f8fafc;
                border: 1px solid #0f172a;
                border-radius: 12px;
                font-weight: 600;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: #1e293b;
                border-color: #1e293b;
                color: #f8fafc;
            }

            [data-testid="stSidebar"] .stRadio,
            [data-testid="stSidebar"] .stSlider,
            [data-testid="stSidebar"] .stFileUploader,
            [data-testid="stSidebar"] .stButton {
                margin-bottom: 0.6rem;
            }

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] .stMarkdown {
                color: #0f172a !important;
            }

            [data-testid="stSidebar"] .stInfo {
                border-radius: 16px;
                border: 1px solid #dbeafe;
            }

            div[data-testid="column"] .stButton > button {
                width: 100%;
                min-height: 3rem;
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 14px;
                font-weight: 700;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
            }

            div[data-testid="column"] .stButton > button:hover {
                background: #eff6ff;
                color: #0f172a;
                border-color: #93c5fd;
            }

            div[data-testid="column"] .stButton > button:focus-visible {
                outline: 3px solid rgba(59, 130, 246, 0.35);
                outline-offset: 2px;
            }

            .quick-action-title {
                font-size: 0.9rem;
                line-height: 1.1;
            }

            .quick-action-icon {
                font-size: 1rem;
                margin-right: 0.4rem;
            }

            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #f8fafc;
                border-radius: 24px;
                padding: 1.6rem 1.7rem;
                margin-bottom: 1rem;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.16);
            }

            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: #93c5fd;
                margin-bottom: 0.65rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.05rem;
                line-height: 1.12;
            }

            .hero p {
                margin: 0.8rem 0 0;
                max-width: 70ch;
                color: #cbd5e1;
                font-size: 0.98rem;
            }

            .status-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin-top: 0.95rem;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                background: rgba(59, 130, 246, 0.12);
                color: #dbeafe;
                border: 1px solid rgba(96, 165, 250, 0.25);
                font-size: 0.88rem;
            }

            .panel {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 20px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                padding: 1rem 1.05rem;
            }

            .chunk-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin: 0.75rem 0;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }

            .chunk-card__header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 650;
                color: #0f172a;
                margin-bottom: 0.45rem;
            }

            .chunk-card__meta {
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 0.55rem;
            }

            .chunk-card__body {
                color: #334155;
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .stChatMessage {
                border-radius: 18px;
            }

            [data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 0.85rem 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Document Studio")
        st.caption("1. Pick a document. 2. Build the index. 3. Ask a question.")

        source_mode = st.radio(
            "Source",
            ["Sample document", "Upload a text file"],
            index=0,
            label_visibility="visible",
        )
        uploaded_file = None
        if source_mode == "Upload a text file":
            uploaded_file = st.file_uploader("Upload", type=["txt", "md"])

        chunk_size = st.slider("Chunk size", min_value=200, max_value=1200, value=500, step=50)
        overlap_max = max(0, chunk_size - 1)
        overlap_default = min(50, overlap_max)
        overlap = st.slider("Chunk overlap", min_value=0, max_value=overlap_max, value=overlap_default, step=10)
        top_k = st.slider("Top retrieved chunks", min_value=1, max_value=5, value=3, step=1)

        st.divider()

        build_clicked = st.button("Build knowledge base", use_container_width=True)
        use_sample_clicked = st.button("Load sample document", use_container_width=True)

        st.info("This app reads a document, retrieves the most relevant chunks, and asks Ollama to answer from that context.")

        st.markdown(
            """
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:0.9rem 1rem;margin-top:0.8rem;">
                <div style="font-weight:700;margin-bottom:0.35rem;color:#0f172a;">Quick start</div>
                <div style="font-size:0.92rem;line-height:1.45;color:#334155;">
                    Use the sample document for a demo, or upload a text file and rebuild the index.
                    Then ask a question like <strong>What is RAG?</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">TinyRAG studio</div>
            <h1>Readable RAG results with visible sources.</h1>
            <p>{APP_TAGLINE}</p>
            <div class="status-chip">The answer, retrieved chunks, and settings stay easy to inspect.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How to use it")
    step_left, step_mid, step_right = st.columns(3)
    with step_left:
        st.markdown("**1. Load a document**\n\nUse the sample document or upload your own `.txt` / `.md` file.")
    with step_mid:
        st.markdown("**2. Build the index**\n\nThe app splits the text into chunks and creates embeddings.")
    with step_right:
        st.markdown("**3. Ask a question**\n\nThe answer appears with the chunks it used below it.")

    if use_sample_clicked:
        build_index(load_sample_text(), "sample_doc.txt", chunk_size, overlap)

    if build_clicked:
        if source_mode == "Upload a text file":
            if uploaded_file is None:
                st.warning("Upload a .txt or .md file before building the index.")
            else:
                text = uploaded_file.getvalue().decode("utf-8")
                build_index(text, uploaded_file.name, chunk_size, overlap)
        else:
            build_index(load_sample_text(), "sample_doc.txt", chunk_size, overlap)

    if st.session_state.rag is None:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.info("Build the index from the sidebar to start asking questions. The sample document is ready if you want a quick demo.")
        st.markdown("Use the sidebar buttons first, then try the example questions below once the index is built.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    render_metrics()
    st.caption(f"Active document: {st.session_state.document_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Chat")
    st.write("Ask a question in plain language. If you are not sure what to ask, try one of the examples below.")

    example_left, example_mid, example_right = st.columns(3)
    with example_left:
        if st.button("🧠 What is RAG?", use_container_width=True):
            ask_and_store("What is RAG?", top_k)
            st.rerun()
    with example_mid:
        if st.button("✂️ Why does chunking matter?", use_container_width=True):
            ask_and_store("Why does chunking matter?", top_k)
            st.rerun()
    with example_right:
        if st.button("🔎 How does retrieval work?", use_container_width=True):
            ask_and_store("How does retrieval work?", top_k)
            st.rerun()

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input("Ask a question about the document")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                answer, sources, error_detail = get_answer(prompt, top_k)

            st.markdown(answer)
            if error_detail:
                st.warning("The app hit a backend issue. The message below is only for troubleshooting.")
                st.code(error_detail)
            elif sources:
                render_sources(sources, st.session_state.document_name or "document")

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources or []})


if __name__ == "__main__":
    main()