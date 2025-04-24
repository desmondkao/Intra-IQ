"""Retrieval module for IntraIQ."""

from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from intraiq.config import DEFAULT_EMBEDDING_MODEL
from intraiq.vector_store import QdrantStore


class DocumentRetriever:
    """Retrieves relevant document chunks from Qdrant."""
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: Optional[str] = None,
        top_k: int = 5
    ):
        """Initialize the document retriever.
        
        Args:
            embedding_model: Model to use for embedding queries
            collection_name: Name of the Qdrant collection to search
            top_k: Number of chunks to retrieve
        """
        self.top_k = top_k
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_store = QdrantStore(collection_name=collection_name)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search Qdrant for similar document chunks
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=self.top_k
        )
        
        return results


class LLMAnswerer:
    """Generates answers using a Hugging Face LLM."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = None,
        max_new_tokens: int = 512
    ):
        """Initialize the LLM answerer.
        
        Args:
            model_name: Hugging Face model to use
            device: Device to run the model on (None for auto-detection)
            max_new_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer for the query using the provided context.
        
        Args:
            query: User question
            context: Context from retrieved documents
            
        Returns:
            Generated answer
        """
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


def run_rag(
    query: str,
    retriever: Optional[DocumentRetriever] = None,
    answerer: Optional[LLMAnswerer] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query.
    
    Args:
        query: User question
        retriever: Optional pre-configured DocumentRetriever
        answerer: Optional pre-configured LLMAnswerer
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary with answer and sources
    """
    # Create retriever if not provided
    if retriever is None:
        retriever = DocumentRetriever(top_k=top_k)
    
    # Retrieve relevant document chunks
    retrieved_docs = retriever.retrieve(query)
    
    # If no documents found, return early
    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": []
        }
    
    # Format context from retrieved documents
    context = "\n\n".join([
        f"Document: {doc['document_name']}\n{doc['text']}"
        for doc in retrieved_docs
    ])
    
    # Create answerer if not provided
    if answerer is None:
        answerer = LLMAnswerer()
    
    # Generate answer
    answer = answerer.generate_answer(query, context)
    
    # Return answer and sources
    return {
        "answer": answer,
        "sources": retrieved_docs
    }