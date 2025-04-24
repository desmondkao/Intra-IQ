"""
IntraIQ Standalone RAG Pipeline

This is a simplified standalone version of IntraIQ that combines all functionality
into a single file to avoid import and configuration issues.

Usage:
1. First run to ingest documents:
   python3 intraiq_standalone.py --ingest --input_dir your_documents/

2. Then run to query:
   python3 intraiq_standalone.py --query "Your question about the documents?"
"""

import os
import re
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
import warnings
import uuid
from transformers import AutoTokenizer



# Silence warnings
warnings.filterwarnings("ignore")

# Try to work around SSL issues
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Document processing
try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    print("Warning: PyMuPDF not installed. PDF support will be limited.")
    fitz = None

# Try to load NLTK, fallback to simple sentence splitting if not available
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    # Try to download nltk data, but don't fail if it doesn't work
    try:
        nltk.download('punkt', quiet=True)
        NLTK_AVAILABLE = True
    except:
        print("Warning: Could not download NLTK data. Using simple tokenization.")
        NLTK_AVAILABLE = False
except ImportError:
    print("Warning: NLTK not installed. Using simple tokenization.")
    NLTK_AVAILABLE = False

# Simple tokenizers as fallback
def simple_sent_tokenize(text):
    """Simple sentence tokenizer that splits on periods, question marks, and exclamation points."""
    return re.split(r'(?<=[.!?])\s+', text)

def simple_word_tokenize(text):
    """Simple word tokenizer that splits on whitespace and punctuation."""
    return re.findall(r'\w+', text)

# Use NLTK if available, otherwise use simple tokenizers
sent_tokenize = simple_sent_tokenize
word_tokenize = simple_word_tokenize
print("Using simple tokenizers.")

# Try to load sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Embedding will not work.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
# Try to load Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    print("Warning: qdrant-client not installed. Vector storage will not work.")
    QDRANT_AVAILABLE = False

# Try to load transformers for LLM
try:
    from transformers import AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. LLM answering will not work.")
    TRANSFORMERS_AVAILABLE = False

# Configuration
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-small"
DEFAULT_TOP_K = 3
DEFAULT_COLLECTION = "intraiq_documents"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QDRANT_LOCAL_PATH = os.path.join(DATA_DIR, "qdrant")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)


class DocumentProcessor:
    """Handles document reading and chunking functionality."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """Initialize the document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required to read PDF files")
            
        doc = fitz.open(file_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
            
        return text
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from a text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def read_document(self, file_path: str) -> str:
        """Read a document file (PDF or text)."""
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext.lower() == '.pdf':
            return self.read_pdf(file_path)
        else:  # Assume it's a text file
            return self.read_text_file(file_path)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens with overlap."""
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Tokenize sentence into words
            tokens = word_tokenize(sentence)
            token_count = len(tokens)
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # finalize the current chunk and start a new one
            if current_token_count + token_count > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # If we have chunk overlap, keep some sentences for the next chunk
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Calculate how many sentences to keep based on token count
                    overlap_tokens = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        s_tokens = len(word_tokenize(s))
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    
                    # Start the next chunk with the overlap sentences
                    current_chunk = overlap_sentences
                    current_token_count = overlap_tokens
                else:
                    # No overlap, start with an empty chunk
                    current_chunk = []
                    current_token_count = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_token_count += token_count
        
        # Add the final chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document: read and chunk."""
        # Get document filename for metadata
        filename = os.path.basename(file_path)
        document_id = re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(filename)[0])
        
        # Read the document
        text = self.read_document(file_path)
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        return {
            "document_id": document_id,
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }


class DocumentEmbedder:
    """Handles embedding generation for document chunks."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize with a SentenceTransformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for embedding")
            
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        return self.model.encode(chunks).tolist()
    
    def embed_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Add embeddings to a processed document."""
        chunks = document["chunks"]
        embeddings = self.get_embeddings(chunks)
        
        # Add embeddings to the document dict
        document["embeddings"] = embeddings
        document["embedding_model"] = self.model_name
        document["embedding_dim"] = self.embedding_dim
        
        return document


class QdrantStore:
    """Handles storage and retrieval of document embeddings in Qdrant."""
    
    def __init__(
        self, 
        collection_name: str = DEFAULT_COLLECTION,
        embedding_dim: Optional[int] = None,
        local_path: str = QDRANT_LOCAL_PATH
    ):
        """Initialize a Qdrant client and ensure collection exists."""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for vector storage")
            
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client
        self.client = QdrantClient(path=local_path)
    
    def create_collection(self, embedding_dim: int) -> None:
        """Create a collection with the specified embedding dimension."""
        self.embedding_dim = embedding_dim
        
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
        else:
            # Check if dimensions match
            collection_info = self.client.get_collection(self.collection_name)
            existing_dim = collection_info.config.params.vectors.size
            
            if existing_dim != embedding_dim:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists with dimension {existing_dim}, "
                    f"but requested dimension is {embedding_dim}."
                )
    
    def store_document(self, document: Dict[str, Any]) -> bool:
        """Store a document with its chunks and embeddings in Qdrant."""
        # Extract document info
        document_id = document["document_id"]
        filename = document["filename"]
        chunks = document["chunks"]
        embeddings = document["embeddings"]
        embedding_dim = document["embedding_dim"]
        
        # Create or verify collection
        if not self.embedding_dim:
            self.create_collection(embedding_dim)
        elif self.embedding_dim != embedding_dim:
            raise ValueError(
                f"Store initialized with dimension {self.embedding_dim}, "
                f"but document has dimension {embedding_dim}."
            )
        
        # Prepare points to add to the collection
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique ID for each chunk
            point_id = str(uuid.uuid4())  # Generate a random UUID

            
            # Create a point with embedding and metadata
            point = rest.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "document_name": filename,
                    "chunk_index": i,
                    "text": chunk
                }
            )
            
            points.append(point)
        
        # Store points in the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Stored {len(points)} chunks for document: {filename}")
        return True
    
    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload["text"],
                "document_id": hit.payload["document_id"],
                "document_name": hit.payload["document_name"],
                "chunk_index": hit.payload["chunk_index"],
                "score": hit.score
            }
            for hit in results
        ]


class LLMAnswerer:
    """Generates answers using a Hugging Face LLM."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        device: str = None,
        max_new_tokens: int = 512
    ):
        """Initialize the LLM answerer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for LLM answering")
            
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading LLM model {model_name} on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        print(f"LLM model loaded")
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer for the query using the provided context."""
        # Create prompt with context and query
        prompt = f"""<s>[INST] You are a helpful AI assistant that answers questions based on the provided context.
Context:
{context}

Question: {query}

Answer the question based on the provided context only. If the context doesn't contain the information needed to answer the question, say so.
[/INST]"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's reply (remove the prompt)
        response = response.split("[/INST]")[-1].strip()
        
        return response


class IntraIQPipeline:
    """Main RAG pipeline for IntraIQ."""
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
        top_k: int = DEFAULT_TOP_K
    ):
        """Initialize the RAG pipeline."""
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize components as needed
        self._processor = None
        self._embedder = None
        self._store = None
        self._llm = None
    
    @property
    def processor(self):
        if self._processor is None:
            self._processor = DocumentProcessor()
        return self._processor
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = DocumentEmbedder(model_name=self.embedding_model)
        return self._embedder
    
    @property
    def store(self):
        if self._store is None:
            self._store = QdrantStore(collection_name=self.collection_name)
        return self._store
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMAnswerer(model_name=self.llm_model)
        return self._llm
    
    def ingest_documents(self, input_dir: str) -> None:
        """Process documents and store them in Qdrant."""
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
            document = self.processor.process_document(file_path)
            print(f"  - Extracted {document['chunk_count']} chunks")
            
            # Generate embeddings
            document = self.embedder.embed_document(document)
            print(f"  - Generated embeddings (dim={document['embedding_dim']})")
            
            # Store in Qdrant
            self.store.store_document(document)
            print(f"  - Stored in Qdrant collection: {self.store.collection_name}")
        
        print("Document processing complete!")
    
    def run_rag(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline to answer a query."""
        print(f"Processing query: {query}")
        
        # Generate embedding for the query
        query_embedding = self.embedder.get_embeddings([query])[0]
        
        # Retrieve relevant document chunks
        print(f"Retrieving top {self.top_k} relevant document chunks...")
        retrieved_docs = self.store.search(query_vector=query_embedding, limit=self.top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        print(f"Retrieved {len(retrieved_docs)} document chunks")
        
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc['document_name']}\n{doc['text']}"
            for doc in retrieved_docs
        ])
        
        # Generate answer
        print("Generating answer...")
        answer = self.llm.generate_answer(query, context)
        
        # Return answer and sources
        return {
            "answer": answer,
            "sources": retrieved_docs
        }


def format_results(results: Dict[str, Any], show_sources: bool = True) -> str:
    """Format results for display."""
    output = f"Answer: {results['answer']}\n"
    
    if show_sources and results['sources']:
        output += "\nSources:\n"
        for i, source in enumerate(results['sources'], 1):
            output += f"\n{i}. Document: {source['document_name']}"
            output += f"\n   Score: {source['score']:.4f}"
            if len(source['text']) > 200:
                preview = source['text'][:200] + "..."
            else:
                preview = source['text']
            output += f"\n   Preview: {preview}\n"
    
    return output


def main():
    """Parse command line arguments and run the IntraIQ pipeline."""
    parser = argparse.ArgumentParser(description="IntraIQ RAG Pipeline")
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--ingest", action="store_true", help="Ingest documents")
    mode_group.add_argument("--query", type=str, help="Query to answer")
    
    # Other arguments
    parser.add_argument("--input_dir", help="Directory containing input documents")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, 
                        help="Number of document chunks to retrieve")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Name of the Qdrant collection")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL,
                        help="Name of the SentenceTransformer model")
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL,
                        help="Hugging Face model to use")
    parser.add_argument("--json", action="store_true", 
                        help="Output results as JSON")
    parser.add_argument("--no-sources", action="store_true",
                        help="Hide source documents in output")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IntraIQPipeline(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        collection_name=args.collection,
        top_k=args.top_k
    )
    
    # Run in the appropriate mode
    if args.ingest:
        if not args.input_dir:
            parser.error("--input_dir is required for ingest mode")
        
        pipeline.ingest_documents(args.input_dir)
    else:  # Query mode
        results = pipeline.run_rag(args.query)
        
        # Output results
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(format_results(results, show_sources=not args.no_sources))


if __name__ == "__main__":
    main()