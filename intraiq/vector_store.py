"""Qdrant vector store integration for IntraIQ."""

from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams

from intraiq.config import (
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY,
    DEFAULT_COLLECTION,
)


class QdrantStore:
    """Handles storage and retrieval of document embeddings in Qdrant."""
    
    def __init__(
        self, 
        collection_name: str = DEFAULT_COLLECTION,
        embedding_dim: Optional[int] = None,
        url: Optional[str] = QDRANT_URL,
        api_key: Optional[str] = QDRANT_API_KEY,
        local_path: Optional[str] = QDRANT_LOCAL_PATH
    ):
        """Initialize a Qdrant client and ensure collection exists.
        
        Args:
            collection_name: Name of the collection to use
            embedding_dim: Dimension of the embedding vectors
            url: URL of Qdrant server (for cloud)
            api_key: API key for Qdrant cloud
            local_path: Path for local Qdrant storage
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client (local or cloud)
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=local_path)
    
    def create_collection(self, embedding_dim: int) -> None:
        """Create a collection with the specified embedding dimension.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
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
        """Store a document with its chunks and embeddings in Qdrant.
        
        Args:
            document: Document dict with chunks and embeddings
            
        Returns:
            True if successful
        """
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
            point_id = f"{document_id}_{i}"
            
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
        """Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of matching points with payload and similarity score
        """
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