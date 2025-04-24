import streamlit as st
import tempfile
import os
from intraiq_standalone import IntraIQPipeline, DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL

st.set_page_config(page_title="IntraIQ", layout="wide")
st.title("🧠 Intra_IQ – Connect, Embed, Query")

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = IntraIQPipeline()

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    embedding_model = st.selectbox("Embedding Model", [DEFAULT_EMBEDDING_MODEL], disabled=True)
    llm_model = st.selectbox("LLM Model", [DEFAULT_LLM_MODEL], disabled=True)
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, 3)

st.session_state.pipeline.top_k = top_k

# --- Upload Section ---
st.subheader("📄 Upload Documents")
uploaded_files = st.file_uploader("Upload .pdf or .txt files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.pipeline.ingest_documents(temp_dir)
    st.success(f"Uploaded and processed {len(uploaded_files)} documents!")

# --- QA Section ---
st.subheader("❓ Ask a Question About Your Documents")
query = st.text_input("Type your question here:")

if query:
    with st.spinner("Thinking..."):
        result = st.session_state.pipeline.run_rag(query)
        st.markdown("### 💬 Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.markdown("### 📚 Sources")
            for i, src in enumerate(result["sources"], 1):
                with st.expander(f"Source {i} – {src['document_name']} (Score: {src['score']:.4f})"):
                    st.write(src["text"][:1000] + "..." if len(src["text"]) > 1000 else src["text"])
