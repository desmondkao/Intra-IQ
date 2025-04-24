# IntraIQ

A domain-adaptable internal document chatbot that uses RAG (retrieval-augmented generation), Hugging Face models, and a LangChain-style agent.

## Features

- Upload and process documents (PDF or text)
- Embed documents and store them in Qdrant
- Use retriever + LLM (Hugging Face) pipeline to answer queries
- Incorporate a LangChain-style agent to select the RAG tool
- Support fine-tuning a Hugging Face model on custom Q&A data
- Hosted via Streamlit

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/intraiq.git
cd intraiq

# Install dependencies
pip install -r requirements.txt
```

## Phase 1: Document Processing and Vector Storage

The first phase focuses on:
- Document ingestion (PDF and text files)
- Text chunking into 500-token segments
- Embedding generation using sentence-transformers
- Storage in Qdrant vector database

### Usage

Process documents and store them in Qdrant:

```bash
python scripts/ingest_documents.py --input_dir /path/to/your/documents
```

Additional options:
```
--chunk_size        Target chunk size in tokens (default: 500)
--chunk_overlap     Number of tokens to overlap between chunks (default: 50)
--embedding_model   Name of the SentenceTransformer model (default: all-MiniLM-L6-v2)
--collection_name   Name of the Qdrant collection
--qdrant_url        URL of Qdrant server (for cloud)
--qdrant_api_key    API key for Qdrant cloud
--qdrant_local_path Path for local Qdrant storage
```

## Project Structure

```
intraiq/                  # Root project directory
├── README.md
├── requirements.txt
├── intraiq/              # Main package
│   ├── __init__.py
│   ├── config.py
│   ├── document_processing.py
│   ├── embeddings.py
│   ├── vector_store.py
├── scripts/
│   └── ingest_documents.py
```

## Next Steps

- Phase 2: Query handling with RAG pipeline
- Phase 3: LangChain-style agent integration
- Phase 4: Hugging Face fine-tuning
- Phase 5: Streamlit UI