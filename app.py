from __future__ import annotations

import streamlit as st

# Schemas are lightweight (pydantic only) — safe to import at module level so
# the sidebar can render before the heavy pipeline modules are loaded.
from src.schemas import ChunkingStrategy, RAGSettings


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 RAG-Powered Knowledge Assistant")
st.caption("Upload one or more .txt files, then ask questions grounded in those files.")

# ---------------------------------------------------------------------------
# Sidebar — rendered immediately; no heavy dependencies needed here
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=list(ChunkingStrategy),
        format_func=lambda s: s.label,
        index=list(ChunkingStrategy).index(ChunkingStrategy.WORD_BOUNDARY),
    )

    chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=2500, value=900, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=800, value=180, step=20)
    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=10, value=4, step=1)

    embedding_model = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )

    st.divider()
    st.markdown(
        "This app uses local semantic retrieval and source-grounded answer synthesis. "
        "Uploaded documents are kept in the current Streamlit session."
    )

# ---------------------------------------------------------------------------
# Heavy modules — imported once and cached for the session lifetime.
# The spinner below is shown during the first load only.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_modules():
    from src.document_loader import load_uploaded_txt_files
    from src.rag_pipeline import StreamlitRAGPipeline
    return load_uploaded_txt_files, StreamlitRAGPipeline


_loading = "app_ready" not in st.session_state

if _loading:
    with st.status("Preparing application…", expanded=True) as _status:
        st.write("Loading embedding model and pipeline dependencies...")
        load_uploaded_txt_files, StreamlitRAGPipeline = _load_modules()
        st.write("Ready.")
        _status.update(label="Application ready!", state="complete", expanded=False)
    st.session_state.app_ready = True
else:
    load_uploaded_txt_files, StreamlitRAGPipeline = _load_modules()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload .txt files",
    type=["txt"],
    accept_multiple_files=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

index_button = st.button("Index uploaded files", type="primary", disabled=not uploaded_files)

if index_button:
    try:
        settings = RAGSettings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
        )
        documents = load_uploaded_txt_files(uploaded_files)
        pipeline = StreamlitRAGPipeline(documents=documents, settings=settings)

        st.session_state.pipeline = pipeline
        st.session_state.indexed_files = [document.filename for document in documents]
        st.session_state.messages = []

        st.success(
            f"Indexed {len(documents)} file(s) into {len(pipeline.chunks)} chunks "
            f"using the **{chunking_strategy.label.split(' —')[0]}** strategy."
        )

    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

if st.session_state.indexed_files:
    with st.expander("Indexed files", expanded=False):
        for filename in st.session_state.indexed_files:
            st.write(f"- {filename}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the uploaded documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.pipeline is None:
        response_text = "Please upload and index at least one .txt file first."
        with st.chat_message("assistant"):
            st.warning(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                try:
                    result = st.session_state.pipeline.ask(question)

                    st.markdown(result.answer)

                    if result.sources:
                        st.markdown("### Sources")
                        for source in result.sources:
                            st.markdown(
                                f"- `{source.filename}` "
                                f"({source.retrieved_chunks} retrieved chunk(s))"
                            )

                    with st.expander("Retrieved context"):
                        for idx, chunk in enumerate(result.retrieved_context, start=1):
                            st.markdown(
                                f"**Chunk {idx}** — `{chunk.filename}` "
                                f"| similarity score: `{chunk.score:.4f}`"
                            )
                            st.write(chunk.text)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.answer}
                    )

                except Exception as exc:
                    error_message = f"Answer generation failed: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
