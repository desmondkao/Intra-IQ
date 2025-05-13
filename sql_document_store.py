"""
SQLDocumentStore - SQL-based document storage for IntraIQ

This module provides SQL database functionality for storing and retrieving 
document metadata and content, which complements the vector-based search.
The implementation uses SQLite for simplicity but follows best practices 
that would scale to enterprise databases.
"""

import sqlite3
import json
from datetime import datetime
import os
import uuid
import pandas as pd

class SQLDocumentStore:
    """
    SQL-based document store using SQLite.
    
    This class handles:
    1. Storing document metadata and content in SQL tables
    2. Basic text search functionality
    3. Document analytics and statistics
    4. Search history tracking
    5. Binary document storage for persistence across sessions
    
    For an interview, you can explain how this would scale to PostgreSQL/SQL Server
    in a production environment.
    """
    
    def __init__(self, db_path="intraiq_documents.db"):
        """
        Initialize the document store with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
        print(f"SQL Document Store initialized with database: {db_path}")
    
    def _create_tables(self):
        """
        Create necessary database tables if they don't exist.
        
        Tables:
        - documents: Stores document metadata and binary content
        - chunks: Stores document content chunks
        - search_history: Tracks user search queries
        - persistent_files: Tracks which documents are stored for persistence
        
        During your interview, you can explain the normalized schema design
        and how foreign keys maintain data integrity.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table for metadata and binary storage
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            document_type TEXT NOT NULL,
            binary_data BLOB,  -- Store the actual file data
            text_content TEXT,
            embedding_model TEXT,
            last_accessed TEXT
        )
        ''')
        
        # Chunks table for document content
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            embedding_id TEXT,
            embedding BLOB,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
        ''')
        
        # Search history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            search_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            document_ids TEXT,
            chunk_ids TEXT,
            enhanced_with_current_info INTEGER DEFAULT 0,
            model_used TEXT,
            document_date_estimate TEXT
        )
        ''')
        
        # Add table for persistent file storage tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persistent_files (
            file_id TEXT PRIMARY KEY,
            document_id TEXT,
            filename TEXT NOT NULL,
            file_size INTEGER,
            upload_date TEXT,
            last_accessed TEXT,
            access_count INTEGER DEFAULT 0,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_document(self, document):
        """
        Store a document and its chunks in the database.
        
        Args:
            document: Document dictionary containing metadata and chunks
            
        Returns:
            Dictionary with status information
            
        During your interview, explain how this method demonstrates:
        1. Proper transaction handling
        2. Parameterized queries for SQL injection prevention
        3. Document/chunk relationship modeling
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Store document metadata
            document_id = document["document_id"]
            filename = document["filename"]
            upload_date = datetime.now().isoformat()
            chunk_count = document["chunk_count"]
            file_size = len(document["text"])
            document_type = filename.split('.')[-1].lower()
            embedding_model = document.get("embedding_model", None)
            
            cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (document_id, filename, upload_date, chunk_count, file_size, document_type, 
             text_content, embedding_model, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (document_id, filename, upload_date, chunk_count, file_size, document_type,
                 document["text"], embedding_model, upload_date))
            
            # Store document chunks
            for i, chunk in enumerate(document["chunks"]):
                chunk_id = f"{document_id}_{i}"
                word_count = len(chunk.split())
                embedding_id = f"{document_id}_{i}" if "embeddings" in document else None
                embedding_data = None
                
                # Store the embedding if available
                if "embeddings" in document and i < len(document["embeddings"]):
                    embedding_data = json.dumps(document["embeddings"][i])
                
                cursor.execute('''
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, chunk_index, text, word_count, embedding_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (chunk_id, document_id, i, chunk, word_count, embedding_id, embedding_data))
            
            # Commit transaction
            conn.commit()
            
            return {
                "document_id": document_id,
                "chunks_stored": len(document["chunks"]),
                "status": "success"
            }
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            conn.close()
    
    def store_binary_document(self, document_id, filename, binary_data):
        """Store binary document data in the SQL database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First check if document exists
            cursor.execute("SELECT document_id FROM documents WHERE document_id = ?", (document_id,))
            result = cursor.fetchone()
            
            if result:
                # Update existing document with binary data
                cursor.execute('''
                UPDATE documents SET binary_data = ?, last_accessed = ? 
                WHERE document_id = ?
                ''', (binary_data, datetime.now().isoformat(), document_id))
            else:
                # Document doesn't exist in the DB yet
                return False
                
            # Add entry to persistent_files table
            file_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT OR REPLACE INTO persistent_files 
            (file_id, document_id, filename, file_size, upload_date, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (
                file_id, 
                document_id, 
                filename, 
                len(binary_data), 
                datetime.now().isoformat(), 
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error storing binary document: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_binary_document(self, document_id):
        """Retrieve binary document data from the SQL database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT document_id, filename, binary_data, document_type 
            FROM documents WHERE document_id = ?
            ''', (document_id,))
            
            result = cursor.fetchone()
            
            if result:
                # Update last_accessed and access_count
                cursor.execute('''
                UPDATE persistent_files 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE document_id = ?
                ''', (datetime.now().isoformat(), document_id))
                
                # Update last_accessed in documents
                cursor.execute('''
                UPDATE documents
                SET last_accessed = ?
                WHERE document_id = ?
                ''', (datetime.now().isoformat(), document_id))
                
                conn.commit()
                
                return {
                    "document_id": result[0],
                    "filename": result[1],
                    "binary_data": result[2],
                    "document_type": result[3]
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error retrieving binary document: {str(e)}")
            return None
        finally:
            conn.close()
            
    def get_all_stored_documents(self):
        """Get a list of all persistently stored documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT p.file_id, p.document_id, p.filename, p.file_size, 
                   p.upload_date, p.access_count, d.document_type
            FROM persistent_files p
            JOIN documents d ON p.document_id = d.document_id
            ORDER BY p.upload_date DESC
            ''')
            
            results = cursor.fetchall()
            
            stored_documents = []
            for row in results:
                stored_documents.append({
                    "file_id": row[0],
                    "document_id": row[1],
                    "filename": row[2],
                    "file_size": row[3],
                    "upload_date": row[4],
                    "access_count": row[5],
                    "document_type": row[6]
                })
                
            return stored_documents
                
        except Exception as e:
            print(f"Error retrieving stored documents: {str(e)}")
            return []
        finally:
            conn.close()
            
    def delete_stored_document(self, document_id):
        """Delete a stored document but keep its metadata and chunks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Remove binary data but keep document record
            cursor.execute('''
            UPDATE documents SET binary_data = NULL
            WHERE document_id = ?
            ''', (document_id,))
            
            # Remove from persistent_files tracking
            cursor.execute('''
            DELETE FROM persistent_files
            WHERE document_id = ?
            ''', (document_id,))
            
            conn.commit()
            return True
                
        except Exception as e:
            print(f"Error deleting stored document: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_document_metadata(self, document_id=None):
        """
        Get metadata for all documents or a specific document.
        
        Args:
            document_id: Optional document ID to retrieve
            
        Returns:
            List of document metadata dictionaries or single document dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if document_id:
                cursor.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    # Remove binary_data from the result to avoid large data transfer
                    doc_dict = dict(zip(columns, result))
                    if 'binary_data' in doc_dict:
                        doc_dict['has_binary_data'] = doc_dict['binary_data'] is not None
                        del doc_dict['binary_data']
                    return doc_dict
                return None
            
            # For all documents, exclude binary data
            cursor.execute("""
            SELECT document_id, filename, upload_date, chunk_count, 
                   file_size, document_type, text_content, embedding_model, 
                   last_accessed, 
                   CASE WHEN binary_data IS NULL THEN 0 ELSE 1 END as has_binary_data
            FROM documents ORDER BY upload_date DESC
            """)
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
        finally:
            conn.close()
    
    def get_document_chunks(self, document_id):
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID to retrieve chunks for
            
        Returns:
            List of chunk dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT chunk_id, document_id, chunk_index, text, word_count, embedding_id
            FROM chunks 
            WHERE document_id = ? 
            ORDER BY chunk_index
            ''', (document_id,))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
        finally:
            conn.close()
    
    def search_documents(self, query_text):
        """
        Perform simple keyword search in document chunks.
        
        Args:
            query_text: Search query text
            
        Returns:
            List of matching chunk dictionaries
            
        During interview, explain the contrast between this SQL full-text search
        and the vector-based semantic search, and how they complement each other.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Use SQL LIKE for basic text search
            search_pattern = f"%{query_text}%"
            cursor.execute('''
            SELECT c.chunk_id, c.document_id, d.filename, c.chunk_index, c.text, c.word_count
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.text LIKE ?
            ORDER BY d.upload_date DESC
            LIMIT 5
            ''', (search_pattern,))
            
            results = cursor.fetchall()
            columns = ['chunk_id', 'document_id', 'filename', 'chunk_index', 'text', 'word_count']
            
            # Log search query
            document_ids = ",".join(set(row[1] for row in results)) if results else ""
            chunk_ids = ",".join(row[0] for row in results) if results else ""
            
            cursor.execute('''
            INSERT INTO search_history (query, timestamp, document_ids, chunk_ids)
            VALUES (?, ?, ?, ?)
            ''', (query_text, datetime.now().isoformat(), document_ids, chunk_ids))
            
            conn.commit()
            
            return [dict(zip(columns, row)) for row in results]
        finally:
            conn.close()
    
    def get_search_history(self, limit=10):
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of search history dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT * FROM search_history
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in results]
        finally:
            conn.close()
    
    def get_document_stats(self):
        """
        Get aggregate statistics about stored documents.
        
        Returns:
            Dictionary with document statistics
            
        During interview, highlight how this demonstrates SQL's strength
        in aggregating and analyzing structured data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get document type distribution
            cursor.execute('''
            SELECT document_type, COUNT(*) as count
            FROM documents
            GROUP BY document_type
            ''')
            doc_types = cursor.fetchall()
            
            # Get total word count
            cursor.execute('SELECT SUM(word_count) FROM chunks')
            total_words = cursor.fetchone()[0] or 0
            
            # Get document count
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count = cursor.fetchone()[0] or 0
            
            # Get chunk count
            cursor.execute('SELECT COUNT(*) FROM chunks')
            chunk_count = cursor.fetchone()[0] or 0
            
            # Get persistent document count
            cursor.execute('SELECT COUNT(*) FROM persistent_files')
            persistent_count = cursor.fetchone()[0] or 0
            
            # Get average chunks per document
            avg_chunks = chunk_count / doc_count if doc_count > 0 else 0
            
            # Get average words per chunk
            cursor.execute('SELECT AVG(word_count) FROM chunks')
            avg_words_per_chunk = cursor.fetchone()[0] or 0
            
            return {
                "document_count": doc_count,
                "persistent_document_count": persistent_count,
                "chunk_count": chunk_count,
                "total_word_count": total_words,
                "avg_chunks_per_document": avg_chunks,
                "avg_words_per_chunk": avg_words_per_chunk,
                "document_types": dict(doc_types)
            }
        finally:
            conn.close()
    
    def run_custom_query(self, query, params=None):
        """
        Run a custom SQL query on the database.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Query results as a list of dictionaries
            
        For interview, this demonstrates the flexibility of SQL for
        custom analytics and advanced queries.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            if results:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return []
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    def debug_tables(self):
        """Print information about database tables and content for debugging."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"Tables in database {self.db_path}: {[t[0] for t in tables]}")
            
            # Check document count
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            print(f"Documents table has {doc_count} rows")
            
            # Check persistent files
            cursor.execute("SELECT COUNT(*) FROM persistent_files")
            persistent_count = cursor.fetchone()[0]
            print(f"Persistent_files table has {persistent_count} rows")
            
            # If there are documents but no persistent files, something is wrong
            if doc_count > 0 and persistent_count == 0:
                print("WARNING: Documents exist but no persistent_files records found!")
                
                # Look at document IDs in the documents table
                cursor.execute("SELECT document_id, filename FROM documents LIMIT 5")
                doc_ids = cursor.fetchall()
                print(f"Sample document_ids in documents table: {doc_ids}")
                
            # Sample documents that have binary data
            cursor.execute("SELECT document_id, filename FROM documents WHERE binary_data IS NOT NULL LIMIT 5")
            binary_docs = cursor.fetchall()
            print(f"Documents with binary data: {binary_docs}")
            
        except Exception as e:
            print(f"Error in debug_tables: {str(e)}")
        finally:
            conn.close()