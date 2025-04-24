#!/usr/bin/env python
"""
Document ingestion script for IntraIQ.

This script processes documents (PDF or text), chunks them,
generates embeddings, and stores them in a Qdrant vector database.
"""

import os
import argparse
from typing import List

from intraiq.document_processing import DocumentProcessor
from intraiq.embeddings import DocumentEmbedder
from intraiq.vector_store import QdrantStore
from intraiq.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_EMBEDDING_MODEL


def ingest_documents(
    input_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = None,
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    qdrant_local_path: str = None
) -> None:
    """Process documents and store them in Qdrant.
    
    Args:
        input_dir: Directory containing input documents
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        embedding_model: Name of the SentenceTransformer model
        collection_name: Name of the Qdrant collection
        qdrant_url: URL of Qdrant server (for cloud)
        qdrant_api_key: API key for Qdrant cloud
        qdrant_local_path: Path for local Qdrant storage
    """
    print(f"Initializing document processor (chunk_size={chunk_size}, overlap={chunk_overlap})")
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print(f"Initializing document embedder (model={embedding_model})")
    embedder = DocumentEmbedder(model_name=embedding_model)
    
    print("Initializing Qdrant vector store")
    vector_store = QdrantStore(
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_api_key,
        local_path=qdrant_local_path
    )
    
    # Get all PDF and text files in the input directory
    files_to_process = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and (file_path.lower().endswith('.pdf') or 
                                          file_path.lower().endswith('.txt')):
            files_to_process.append(file_path)
    
    print(f"Found {len(files_to_process)} documents to process")
    
    # Process each file
    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        print(f"Processing document: {filename}")
        
        # Process the document (read and chunk)
        document = processor.process_document(file_path)
        print(f"  - Extracted {document['chunk_count']} chunks")
        
        # Generate embeddings
        document = embedder.embed_document(document)
        print(f"  - Generated embeddings (dim={document['embedding_dim']})")
        
        # Store in Qdrant
        vector_store.store_document(document)
        print(f"  - Stored in Qdrant collection: {vector_store.collection_name}")
    
    print("Document processing complete!")


def main():
    """Parse command line arguments and run the ingestion process."""
    parser = argparse.ArgumentParser(description="Process documents and store in Qdrant")
    parser.add_argument("--input_dir", required=True, help="Directory containing input documents")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, 
                        help="Target chunk size in tokens")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Number of tokens to overlap between chunks")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL, 
                        help="Name of the SentenceTransformer model")
    parser.add_argument("--collection_name", help="Name of the Qdrant collection")
    parser.add_argument("--qdrant_url", help="URL of Qdrant server (for cloud)")
    parser.add_argument("--qdrant_api_key", help="API key for Qdrant cloud")
    parser.add_argument("--qdrant_local_path", help="Path for local Qdrant storage")
    
    args = parser.parse_args()
    
    ingest_documents(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_local_path=args.qdrant_local_path
    )


if __name__ == "__main__":
    main()