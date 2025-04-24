"""Document processor for IntraIQ."""

import os
import re
from typing import List, Dict, Any

import fitz  # PyMuPDF for PDF processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from intraiq.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Make sure we have the NLTK data we need
nltk.download('punkt', quiet=True)


class DocumentProcessor:
    """Handles document reading and chunking functionality."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """Initialize the document processor.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        doc = fitz.open(file_path)
        text = ""
        
        for page in doc:
            text += page.get_text()
            
        return text
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content as a string
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def read_document(self, file_path: str) -> str:
        """Read a document file (PDF or text).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Text content as a string
        """
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext.lower() == '.pdf':
            return self.read_pdf(file_path)
        else:  # Assume it's a text file
            return self.read_text_file(file_path)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens with overlap.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
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
        """Process a document: read and chunk.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document info and chunks
        """
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