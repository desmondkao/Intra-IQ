import os
import re
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
import warnings
import uuid
import tempfile
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

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
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. LLM answering will not work.")
    TRANSFORMERS_AVAILABLE = False

# Try to load OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai not installed or outdated. OpenAI integration will not work.")
    OPENAI_AVAILABLE = False

# Try to load requests for Tavily API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests not installed. Tavily search integration will not work.")
    REQUESTS_AVAILABLE = False

# Configuration
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-small"
DEFAULT_TOP_K = 3
DEFAULT_COLLECTION = "opus_documents"
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

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the store."""
        try:
            # Find all points with this document_id
            filter_query = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {
                            "value": document_id
                        }
                    }
                ]
            }
            
            # Get points matching the filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_query,
                limit=1000,  # Adjust based on expected max chunks per document
                with_payload=False,
                with_vectors=False
            )
            
            if not search_result.points:
                print(f"No points found for document_id: {document_id}")
                return False
            
            # Extract point IDs
            point_ids = [point.id for point in search_result.points]
            
            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=point_ids
                )
            )
            
            print(f"Deleted {len(point_ids)} chunks for document: {document_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
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
        
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the store."""
        try:
            # Find all points with this document_id
            filter_query = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {
                            "value": document_id
                        }
                    }
                ]
            }
            
            # Get points matching the filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_query,
                limit=1000,  # Adjust based on expected max chunks per document
                with_payload=False,
                with_vectors=False
            )
            
            if not search_result.points:
                print(f"No points found for document_id: {document_id}")
                return False
            
            # Extract point IDs
            point_ids = [point.id for point in search_result.points]
            
            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=point_ids
                )
            )
            
            print(f"Deleted {len(point_ids)} chunks for document: {document_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with special handling for tabular financial data."""
        # First identify potential table sections
        table_pattern = re.compile(r'(\n[\s\d,\.]+\n[\s\d,\.]+\n)')
        
        # Split by potential tables first
        parts = table_pattern.split(text)
        
        chunks = []
        for part in parts:
            # If this part looks like tabular data (lots of numbers and few words)
            if sum(c.isdigit() for c in part) > len(part) * 0.3 and len(re.findall(r'[A-Za-z]{4,}', part)) < 5:
                # Keep table sections together as a single chunk
                if part.strip():  # Only add non-empty chunks
                    chunks.append(part)
            else:
                # Process regular text using the standard method
                sentences = sent_tokenize(part)
                
                current_chunk = []
                current_token_count = 0
                
                for sentence in sentences:
                    tokens = word_tokenize(sentence)
                    token_count = len(tokens)
                    
                    if current_token_count + token_count > self.chunk_size and current_chunk:
                        # Create a chunk identifier to help with uniqueness
                        chunk_text = " ".join(current_chunk)
                        if chunk_text.strip():  # Only add non-empty chunks
                            chunks.append(chunk_text)
                        
                        if self.chunk_overlap > 0 and len(current_chunk) > 1:
                            # Calculate overlap sentences
                            overlap_tokens = 0
                            overlap_sentences = []
                            
                            for s in reversed(current_chunk):
                                s_tokens = len(word_tokenize(s))
                                if overlap_tokens + s_tokens <= self.chunk_overlap:
                                    overlap_sentences.insert(0, s)
                                    overlap_tokens += s_tokens
                                else:
                                    break
                            
                            current_chunk = overlap_sentences
                            current_token_count = overlap_tokens
                        else:
                            current_chunk = []
                            current_token_count = 0
                    
                    current_chunk.append(sentence)
                    current_token_count += token_count
                
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if chunk_text.strip():  # Only add non-empty chunks
                        chunks.append(chunk_text)
        
        # Add chunk index prefix to help with differentiation in embedding space
        return [f"[Chunk {i+1}/{len(chunks)}] {chunk}" for i, chunk in enumerate(chunks)]
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
        local_path: str = None
    ):
        """Initialize a Qdrant client and ensure collection exists."""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for vector storage")
            
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Use a unique path if none provided
        if local_path is None:
            local_path = os.path.join(tempfile.gettempdir(), f"qdrant_{uuid.uuid4()}")
            os.makedirs(local_path, exist_ok=True)
            print(f"Using unique Qdrant path: {local_path}")
        
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
        try:
            input_text = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Generate output
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                no_repeat_ngram_size=3
            )
            
            # Decode the generated output
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return answer
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Modified OpenAIAnswerer Class
class OpenAIAnswerer:
    """Generates answers using OpenAI API with improved prompting."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 512
    ):
        """Initialize the OpenAI answerer with model selection."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAI integration")
            
        # Use environment variable if no API key provided
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        # Support for multiple models with fallback options
        self.available_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
        
        # Set the model with fallback options
        if model in self.available_models:
            self.model = model
        else:
            print(f"Warning: Model '{model}' not recognized. Falling back to gpt-3.5-turbo.")
            self.model = "gpt-3.5-turbo"
            
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer for the query using the provided context with improved prompting."""
        try:
            # Enhanced system prompt with better financial focus and formatting instructions
            system_prompt = """You are an expert financial analyst assistant that provides comprehensive, accurate answers based on the provided document context.

Your responses should:
1. Be authoritative yet conversational - write as a single expert voice
2. Format financial data consistently (e.g., "$5.2 million", "12% growth")
3. Use plain text formatting (no italics or special formatting)
4. Highlight key financial insights and trends
5. Structure information logically with clear paragraphs
6. Maintain a balanced analytical perspective
7. Avoid phrases like "according to the documents" or "based on the context"

Present yourself as having directly analyzed the financial information."""

            # Improved user prompt with context about document nature
            user_prompt = f"""Answer the following question using only the provided financial document context. These documents contain reliable financial information that you've analyzed directly.

Context:
{context}

Question: {query}

Provide a comprehensive analysis that:
- Addresses the specific question directly
- Presents numerical data with proper formatting
- Maintains a cohesive narrative throughout
- Uses clear paragraph structure
- Focuses on the most relevant financial insights"""

            # Call the OpenAI API with the selected model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.2,  # Low temperature for more factual responses
            )
            
            # Extract and return the answer
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating answer with OpenAI: {str(e)}"


# Modified TavilySearchAgent Class
class TavilySearchAgent:
    """An agentic component that uses Tavily to search for current information 
    and seamlessly integrates it with document-based answers."""
    
    def __init__(self, openai_api_key=None, tavily_api_key=None, default_model="gpt-3.5-turbo"):
        """Initialize with API keys and default model."""
        # Check required dependencies
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for the search agent")
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required for the search agent")
            
        # Initialize API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        
        # Set default model with validation
        self.available_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
        if default_model in self.available_models:
            self.default_model = default_model
        else:
            print(f"Warning: Model '{default_model}' not recognized. Falling back to gpt-3.5-turbo.")
            self.default_model = "gpt-3.5-turbo"
        
        # Initialize clients if API keys are available
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("Warning: OpenAI API key not provided. Agent decision-making will be limited.")
        
        self.tavily_endpoint = "https://api.tavily.com/search"
    
    def evaluate_search_need(self, query, document_info, model=None):
        """Use OpenAI to determine if a search is needed based on the query and available document info."""
        if not self.openai_client:
            # Fallback to rule-based method if no OpenAI API key
            return self._rule_based_evaluation(query)
        
        # Use specified model or fall back to default
        model = model if model in self.available_models else self.default_model
        
        # Create a system prompt that explains the agent's role
        system_prompt = """You are an AI assistant helping to determine if a user's financial query requires current market information beyond what's in their documents.
        
    You should return 'should_search: true' ONLY when the query explicitly or implicitly requests:
    1. Current financial data, prices, or market conditions
    2. Comparisons between historical and current information
    3. Recent events, news, or developments
    4. Up-to-date regulatory information
    5. Current market benchmarks or averages

    Return 'should_search: false' when:
    1. The query can be fully answered from the documents
    2. The query is about general concepts, definitions, or historical information
    3. The query doesn't involve time-sensitive financial information
    """

        # Create a user message that includes the query and document info
        user_message = f"""Query: {query}

    Available Document Information:
    {document_info}

    Based on this query and the available documents, determine if current market information from the web is needed.
    """

        # Define the function that OpenAI can call
        functions = [
            {
                "name": "determine_search_need",
                "description": "Determines if a web search for current financial information is necessary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "should_search": {
                            "type": "boolean",
                            "description": "Whether a search for current information is needed"
                        },
                        "search_query": {
                            "type": "string", 
                            "description": "The optimized search query to use if search is needed"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Explanation of why search is or isn't needed"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific aspects to focus on in the search results"
                        },
                        "document_date_estimate": {
                            "type": "string",
                            "description": "Estimated date range or period the documents cover (e.g., '2021-2022', 'Q3 2022')"
                        }
                    },
                    "required": ["should_search", "reason"]
                }
            }
        ]

        # Call OpenAI to evaluate if search is needed
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                functions=functions,
                function_call={"name": "determine_search_need"}
            )
            
            # Extract the function call
            function_call = response.choices[0].message.function_call
            if function_call:
                args = json.loads(function_call.arguments)
                return args
            else:
                return {"should_search": False, "reason": "No function call in response"}
                
        except Exception as e:
            print(f"Error in search evaluation: {str(e)}")
            return {"should_search": False, "reason": f"Error: {str(e)}"}
    
    def _rule_based_evaluation(self, query):
        """Simple rule-based method to determine if search is needed (fallback when no OpenAI key)."""
        # Keywords suggesting need for current information
        current_info_keywords = [
            "current", "latest", "recent", "today", "now", "update",
            "price", "compare", "comparison", "today's", "this month",
            "this year", "this quarter", "market value", "stock price"
        ]
        
        # Check if query contains any of these keywords
        should_search = any(keyword in query.lower() for keyword in current_info_keywords)
        
        if should_search:
            return {
                "should_search": True,
                "search_query": query,
                "reason": "Query appears to request current market information",
                "focus_areas": []
            }
        else:
            return {
                "should_search": False,
                "reason": "Query does not appear to require current information"
            }
    
    def execute_search(self, search_query, focus_areas=None):
        """Execute a search using Tavily API."""
        if not self.tavily_api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured in .env file",
                "results": []
            }
        
        # Add finance-specific context if not already present
        if not any(term in search_query.lower() for term in ["finance", "financial", "market", "stock", "economy"]):
            search_query = f"financial information {search_query}"
        
        # Build search parameters
        domains = ["bloomberg.com", "reuters.com", "ft.com", "wsj.com", "cnbc.com", 
                  "marketwatch.com", "finance.yahoo.com", "economist.com", "forbes.com"]
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.tavily_api_key}"
            }
            
            payload = {
                "query": search_query,
                "search_depth": "advanced",
                "include_domains": domains,
                "max_results": 5,
                "include_answer": False,  # Skip Tavily's generated answer
                "include_raw_content": True,  # Get raw content instead of processed content
                "include_citations": False  # Skip citations which might include markdown
            }
            
            # Add focus areas to query if provided
            if focus_areas and isinstance(focus_areas, list) and len(focus_areas) > 0:
                focus_string = " ".join(focus_areas)
                payload["query"] = f"{search_query} {focus_string}"
            
            # Execute the search
            response = requests.post(
                self.tavily_endpoint,
                headers=headers,
                json=payload
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "Tavily API authentication failed (status 401). Please check your API key.",
                    "response_text": response.text,
                    "results": []
                }
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "results": response.json().get("results", []),
                    "query": payload["query"]
                }
            else:
                return {
                    "success": False,
                    "error": f"Tavily API returned status code {response.status_code}",
                    "response_text": response.text,
                    "results": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def synthesize_information(self, query, rag_answer, search_results, document_date_estimate=None, model=None):
        """Use OpenAI to synthesize RAG answer with search results into a cohesive, unified response."""
        # If no search results, return the original RAG answer
        if not search_results.get("success") or not search_results.get("results"):
            return rag_answer
            
        # If no OpenAI client, fallback to simple concatenation
        if not self.openai_client:
            search_info = "\n\nCurrent Market Information:\n"
            for i, result in enumerate(search_results["results"][:2], 1):
                search_info += f"- {result.get('title', 'Source ' + str(i))}: {result.get('content', 'No content')[:200]}...\n"
            return f"{rag_answer}\n\n{search_info}"
        
        # Use specified model or fall back to default
        model = model if model in self.available_models else self.default_model
        
        # Create context from search results
        search_context = ""
        for i, result in enumerate(search_results["results"][:3], 1):
            search_context += f"\nSource {i}: {result.get('title', 'Unknown')}\n"
            search_context += f"URL: {result.get('url', 'Unknown')}\n"
            
            # Extract and format the date if available
            pub_date = result.get('published_date', 'Unknown')
            if pub_date and pub_date.lower() != 'unknown':
                search_context += f"Date: {pub_date}\n"
            
            search_context += f"Content: {result.get('content', 'No content available')}\n\n"
        
        # Create temporal context string
        temporal_context = ""
        if document_date_estimate and document_date_estimate.lower() != 'unknown':
            temporal_context = f"The financial documents contain information from approximately {document_date_estimate}, while the search results provide current market information as of today."
        
        # Create system prompt with improved instructions for cohesive synthesis
        system_prompt = f"""You are an expert financial analyst synthesizing historical document data with current market information.

Your task is to create a SINGLE, UNIFIED RESPONSE that seamlessly integrates:
1. Information from financial documents ({document_date_estimate if document_date_estimate else "historical data"})
2. Current market information from recent sources

CRITICAL FORMATTING REQUIREMENTS:
- Use plain text only - absolutely NO italics or special formatting
- Present numerical data consistently (e.g., "$50 million" not "$50million")
- Ensure proper spacing throughout (after punctuation, between words)
- Use standard quotation marks when needed
- Maintain clean paragraph structure for readability

YOUR RESPONSE MUST:
- Read as if written by a single financial expert with comprehensive knowledge
- Flow naturally between historical and current information
- NEVER use phrases like "according to the documents" or "based on search results"
- NEVER have separate sections for document vs. search information
- Create explicit connections between historical data and current conditions
- Maintain a consistent analytical voice throughout
- Provide temporal context where relevant without explicitly mentioning document dates

{temporal_context}

Your analysis should reflect deep expertise in financial markets and documentation.
"""

        # Create user message with query and both information sources
        user_message = f"""Original Query: {query}

Historical Financial Information:
{rag_answer}

Current Market Information:
{search_context}

Create a single, cohesive financial analysis that addresses the query by integrating both the historical documentation and current market context. Your response should flow naturally as one unified analysis.
"""

        # Call OpenAI with the selected model
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3  # Lower temperature for more factual synthesis
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            # Return original answer with error note in case of failure
            print(f"Error in synthesis: {str(e)}")
            return f"{rag_answer}\n\n[Note: Attempted to enhance with current market data but encountered an error: {str(e)}]"
    
    def process_query(self, query, rag_answer, rag_sources, model=None):
        """Full pipeline: evaluate search need, execute search if needed, synthesize results."""
        # Use specified model or fall back to default
        model = model if model in self.available_models else self.default_model
        
        # Create document info summary for the agent
        doc_info = "Available documents:\n"
        for i, src in enumerate(rag_sources[:3], 1):
            doc_info += f"- {src.get('document_name', 'Document ' + str(i))}\n"
            doc_info += f"  Preview: {src.get('text', '')[:100]}...\n"
        
        # Step 1: Evaluate if search is needed with model selection
        evaluation = self.evaluate_search_need(query, doc_info, model=model)
        
        # Extract document date estimate if available
        document_date_estimate = evaluation.get("document_date_estimate", "Unknown")
        
        # If search is not needed, return original answer
        if not evaluation.get("should_search", False):
            return {
                "answer": rag_answer,
                "sources": rag_sources,
                "search_performed": False,
                "reasoning": evaluation.get("reason", "Search was deemed unnecessary")
            }
        
        # Step 2: Execute search with the optimized query
        search_query = evaluation.get("search_query", query)
        focus_areas = evaluation.get("focus_areas", [])
        search_results = self.execute_search(search_query, focus_areas)
        
        # If search failed, return original answer with explanation
        if not search_results.get("success", False):
            return {
                "answer": rag_answer,
                "sources": rag_sources,
                "search_performed": True,
                "search_successful": False,
                "search_error": search_results.get("error", "Unknown search error")
            }
        
        # Step 3: Synthesize information with temporal context and model selection
        enhanced_answer = self.synthesize_information(
            query, 
            rag_answer, 
            search_results,
            document_date_estimate=document_date_estimate,
            model=model
        )
        
        # Step 4: Add search results as additional sources
        enhanced_sources = rag_sources + [
            {
                "document_name": result.get("title", "Web Search Result"),
                "text": result.get("content", "No content available"),
                "score": 0.85,  # High relevance for current information
                "source_type": "web_search",
                "url": result.get("url", "#"),
                "published_date": result.get("published_date", "Unknown")
            }
            for result in search_results.get("results", [])[:3]
        ]
        
        # Return enhanced result
        return {
            "answer": enhanced_answer,
            "sources": enhanced_sources,
            "search_performed": True,
            "search_successful": True,
            "search_query": search_results.get("query", search_query),
            "reasoning": evaluation.get("reason", "Search was performed based on query analysis"),
            "search_results": search_results.get("results", []),
            "document_date_estimate": document_date_estimate
        }


# Modifications to IntraIQPipeline class to support model selection
class IntraIQPipeline:
    """Main RAG pipeline for IntraIQ with improved model selection."""

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
            # Use consistent path to ensure persistence
            qdrant_path = os.path.join(DATA_DIR, "qdrant")
            os.makedirs(qdrant_path, exist_ok=True)
            
            self._store = QdrantStore(
                collection_name=self.collection_name,
                local_path=qdrant_path
            )
        return self._store

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMAnswerer(model_name=self.llm_model)
        return self._llm

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
        top_k: int = DEFAULT_TOP_K,
        openai_model: str = "gpt-3.5-turbo"  # Add default OpenAI model parameter
    ):
        """Initialize the RAG pipeline."""
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Set the OpenAI model with validation
        self.available_openai_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
        if openai_model in self.available_openai_models:
            self.openai_model = openai_model
        else:
            print(f"Warning: OpenAI model '{openai_model}' not recognized. Falling back to gpt-3.5-turbo.")
            self.openai_model = "gpt-3.5-turbo"
        
        # Initialize components as needed
        self._processor = None
        self._embedder = None
        self._store = None
        self._llm = None
        self._search_agent = None
    
    def ingest_documents(self, input_dir: str) -> List[str]:
        """
        Process documents and store them in Qdrant.
        Returns a list of document IDs for the processed documents.
        """
        # Get all PDF and text files in the input directory
        files_to_process = []
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path) and (file_path.lower().endswith('.pdf') or 
                                            file_path.lower().endswith('.txt')):
                files_to_process.append(file_path)
        
        print(f"Found {len(files_to_process)} documents to process")
        
        # Track document IDs for return
        document_ids = []
        
        # Process each file
        for file_path in files_to_process:
            filename = os.path.basename(file_path)
            print(f"Processing document: {filename}")
            
            # Process the document (read and chunk)
            document = self.processor.process_document(file_path)
            print(f"  - Extracted {document['chunk_count']} chunks")
            
            # Add document ID to return list
            document_ids.append(document["document_id"])
            
            # Generate embeddings
            document = self.embedder.embed_document(document)
            print(f"  - Generated embeddings (dim={document['embedding_dim']})")
            
            # Store in Qdrant
            self.store.store_document(document)
            print(f"  - Stored in Qdrant collection: {self.store.collection_name}")
        
        print("Document processing complete!")
        return document_ids
    
    @property
    def search_agent(self):
        if self._search_agent is None:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            
            if openai_api_key and tavily_api_key:
                try:
                    self._search_agent = TavilySearchAgent(
                        openai_api_key=openai_api_key,
                        tavily_api_key=tavily_api_key,
                        default_model=self.openai_model  # Pass the selected model
                    )
                except ImportError as e:
                    print(f"Warning: Could not initialize search agent: {str(e)}")
                    self._search_agent = None
            else:
                print("Warning: OpenAI or Tavily API key missing. Search agent will not be available.")
                self._search_agent = None
                    
        return self._search_agent
    
    # Updated run_rag method with model selection
    def run_rag(self, query: str, use_openai: bool = False, openai_api_key: str = None, 
                openai_model: str = None, enable_search_agent: bool = False) -> Dict[str, Any]:
        """Run the RAG pipeline to answer a query with model selection."""
        print(f"Processing query: {query}")
        
        # Use provided model or fall back to instance default
        if openai_model in self.available_openai_models:
            selected_model = openai_model
        else:
            selected_model = self.openai_model
            
        print(f"Using OpenAI model: {selected_model}")
        
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
        
        # Generate answer based on the selected method
        print("Generating answer...")
        
        if use_openai and openai_api_key:
            # Use OpenAI if selected and API key provided
            try:
                openai_answerer = OpenAIAnswerer(
                    api_key=openai_api_key,
                    model=selected_model  # Use the selected model
                )
                answer = openai_answerer.generate_answer(query, context)
            except Exception as e:
                answer = f"Error using OpenAI: {str(e)}. Falling back to local model."
                # Fall back to local model
                answer = self.llm.generate_answer(query, context)
        else:
            # Use local model
            answer = self.llm.generate_answer(query, context)
        
        # Initial result without search enhancement
        result = {
            "answer": answer,
            "sources": retrieved_docs
        }
        
        # Use search agent to enhance with current information if requested and available
        if enable_search_agent and self.search_agent:
            print("Evaluating search needs...")
            enhanced_result = self.search_agent.process_query(
                query, 
                result["answer"], 
                result["sources"],
                model=selected_model  # Pass the selected model
            )
            result.update(enhanced_result)
        
        return result