# SQL Document Store for PostgreSQL
# Modified version to work with PostgreSQL instead of SQLite

import psycopg2
import psycopg2.extras
import json
from datetime import datetime
import os
import uuid
import pandas as pd
import streamlit as st

class PostgreSQLDocumentStore:
    """
    PostgreSQL-based document store.
    
    This class handles:
    1. Storing document metadata and content in PostgreSQL tables
    2. Basic text search functionality
    3. Document analytics and statistics
    4. Search history tracking
    5. Binary document storage for persistence across sessions
    """
    
    def __init__(self, db_path=None, connection_string=None):
        """
        Initialize the document store with database path or connection string.
        
        Args:
            db_path: "postgres" indicator (used for compatibility with old code)
            connection_string: PostgreSQL connection string (not used if using streamlit cached connection)
        """
        self.db_path = db_path  # Keep for compatibility
        self.connection_string = connection_string
        self._create_tables()
        print(f"PostgreSQL Document Store initialized")
    
    def get_connection(self):
        """Get a connection from the Streamlit cached resource or create a new one."""
        if hasattr(st, 'session_state') and 'get_db_connection' in globals():
            # Use the Streamlit cached connection
            return get_db_connection()
        else:
            # If not in Streamlit or connection function not available, create a new connection
            if not self.connection_string:
                # Get PostgreSQL connection parameters from environment variables
                DB_HOST = os.environ.get("DB_HOST", "localhost")
                DB_PORT = os.environ.get("DB_PORT", "5432")
                DB_NAME = os.environ.get("DB_NAME", "hermes_documents")
                DB_USER = os.environ.get("DB_USER", "postgres")
                DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
                
                # Create connection
                conn = psycopg2.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD
                )
                conn.autocommit = False  # Use transactions explicitly
                return conn
            else:
                conn = psycopg2.connect(self.connection_string)
                conn.autocommit = False
                return conn
    
    def _create_tables(self):
        """
        Create necessary database tables if they don't exist.
        
        Tables:
        - documents: Stores document metadata and binary content
        - chunks: Stores document content chunks
        - search_history: Tracks user search queries
        - persistent_files: Tracks which documents are stored for persistence
        """
        conn = self.get_connection()
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
            binary_data BYTEA,  -- Store the actual file data
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
            embedding TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
        ''')
        
        # Search history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            search_id SERIAL PRIMARY KEY,
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
        cursor.close()
    
    def store_document(self, document):
        """
        Store a document and its chunks in the PostgreSQL database.
        
        Args:
            document: Document dictionary containing metadata and chunks
            
        Returns:
            Dictionary with status information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Begin transaction
            cursor.execute("BEGIN")
            
            # Store document metadata
            document_id = document["document_id"]
            filename = document["filename"]
            upload_date = datetime.now().isoformat()
            chunk_count = document["chunk_count"]
            file_size = len(document["text"])
            document_type = filename.split('.')[-1].lower()
            embedding_model = document.get("embedding_model", None)
            
            cursor.execute('''
            INSERT INTO documents 
            (document_id, filename, upload_date, chunk_count, file_size, document_type, 
             text_content, embedding_model, last_accessed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE
            SET filename = EXCLUDED.filename,
                upload_date = EXCLUDED.upload_date,
                chunk_count = EXCLUDED.chunk_count,
                file_size = EXCLUDED.file_size,
                document_type = EXCLUDED.document_type,
                text_content = EXCLUDED.text_content,
                embedding_model = EXCLUDED.embedding_model,
                last_accessed = EXCLUDED.last_accessed
            ''', (
                document_id, filename, upload_date, chunk_count, file_size, document_type,
                document["text"], embedding_model, upload_date
            ))
            
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
                INSERT INTO chunks
                (chunk_id, document_id, chunk_index, text, word_count, embedding_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET document_id = EXCLUDED.document_id,
                    chunk_index = EXCLUDED.chunk_index,
                    text = EXCLUDED.text,
                    word_count = EXCLUDED.word_count,
                    embedding_id = EXCLUDED.embedding_id,
                    embedding = EXCLUDED.embedding
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
            cursor.close()
    
    def store_binary_document(self, document_id, filename, binary_data):
        """Store binary document data in the PostgreSQL database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # First check if document exists
            cursor.execute("SELECT document_id FROM documents WHERE document_id = %s", (document_id,))
            result = cursor.fetchone()
            
            if result:
                # Update existing document with binary data
                cursor.execute('''
                UPDATE documents SET binary_data = %s, last_accessed = %s 
                WHERE document_id = %s
                ''', (psycopg2.Binary(binary_data), datetime.now().isoformat(), document_id))
            else:
                # Document doesn't exist in the DB yet
                return False
                
            # Add entry to persistent_files table
            file_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO persistent_files 
            (file_id, document_id, filename, file_size, upload_date, last_accessed, access_count)
            VALUES (%s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (file_id) DO UPDATE
            SET document_id = EXCLUDED.document_id,
                filename = EXCLUDED.filename,
                file_size = EXCLUDED.file_size,
                upload_date = EXCLUDED.upload_date,
                last_accessed = EXCLUDED.last_accessed,
                access_count = persistent_files.access_count + 1
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
            conn.rollback()
            print(f"Error storing binary document: {str(e)}")
            return False
        finally:
            cursor.close()
    
    def get_binary_document(self, document_id):
        """Retrieve binary document data from the PostgreSQL database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT document_id, filename, binary_data, document_type 
            FROM documents WHERE document_id = %s
            ''', (document_id,))
            
            result = cursor.fetchone()
            
            if result:
                # Update last_accessed and access_count
                cursor.execute('''
                UPDATE persistent_files 
                SET last_accessed = %s, access_count = access_count + 1
                WHERE document_id = %s
                ''', (datetime.now().isoformat(), document_id))
                
                # Update last_accessed in documents
                cursor.execute('''
                UPDATE documents
                SET last_accessed = %s
                WHERE document_id = %s
                ''', (datetime.now().isoformat(), document_id))
                
                conn.commit()
                
                return {
                    "document_id": result[0],
                    "filename": result[1],
                    "binary_data": bytes(result[2]) if result[2] else None,
                    "document_type": result[3]
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error retrieving binary document: {str(e)}")
            return None
        finally:
            cursor.close()
            
    def get_all_stored_documents(self):
        """Get a list of all persistently stored documents."""
        conn = self.get_connection()
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
            cursor.close()
            
    def delete_stored_document(self, document_id):
        """Delete a stored document but keep its metadata and chunks."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Remove binary data but keep document record
            cursor.execute('''
            UPDATE documents SET binary_data = NULL
            WHERE document_id = %s
            ''', (document_id,))
            
            # Remove from persistent_files tracking
            cursor.execute('''
            DELETE FROM persistent_files
            WHERE document_id = %s
            ''', (document_id,))
            
            conn.commit()
            return True
                
        except Exception as e:
            conn.rollback()
            print(f"Error deleting stored document: {str(e)}")
            return False
        finally:
            cursor.close()
    
    def get_document_metadata(self, document_id=None):
        """
        Get metadata for all documents or a specific document.
        
        Args:
            document_id: Optional document ID to retrieve
            
        Returns:
            List of document metadata dictionaries or single document dictionary
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            if document_id:
                cursor.execute("""
                SELECT 
                    document_id, filename, upload_date, chunk_count, 
                    file_size, document_type, text_content, embedding_model, 
                    last_accessed, 
                    CASE WHEN binary_data IS NULL THEN FALSE ELSE TRUE END as has_binary_data
                FROM documents WHERE document_id = %s
                """, (document_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
            
            # For all documents, exclude binary data
            cursor.execute("""
            SELECT 
                document_id, filename, upload_date, chunk_count, 
                file_size, document_type, text_content, embedding_model, 
                last_accessed, 
                CASE WHEN binary_data IS NULL THEN FALSE ELSE TRUE END as has_binary_data
            FROM documents ORDER BY upload_date DESC
            """)
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
    
    def get_document_chunks(self, document_id):
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID to retrieve chunks for
            
        Returns:
            List of chunk dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            cursor.execute('''
            SELECT chunk_id, document_id, chunk_index, text, word_count, embedding_id
            FROM chunks 
            WHERE document_id = %s 
            ORDER BY chunk_index
            ''', (document_id,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
    
    def search_documents(self, query_text):
        """
        Perform simple keyword search in document chunks.
        
        Args:
            query_text: Search query text
            
        Returns:
            List of matching chunk dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            # Use ILIKE for case-insensitive search in PostgreSQL
            search_pattern = f"%{query_text}%"
            cursor.execute('''
            SELECT c.chunk_id, c.document_id, d.filename, c.chunk_index, c.text, c.word_count
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.text ILIKE %s
            ORDER BY d.upload_date DESC
            LIMIT 5
            ''', (search_pattern,))
            
            results = cursor.fetchall()
            
            # Log search query
            document_ids = ",".join([row["document_id"] for row in results]) if results else ""
            chunk_ids = ",".join([row["chunk_id"] for row in results]) if results else ""
            
            cursor.execute('''
            INSERT INTO search_history (query, timestamp, document_ids, chunk_ids)
            VALUES (%s, %s, %s, %s)
            ''', (query_text, datetime.now().isoformat(), document_ids, chunk_ids))
            
            conn.commit()
            
            return [dict(row) for row in results]
        finally:
            cursor.close()
    
    def get_search_history(self, limit=10):
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of search history dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            cursor.execute('''
            SELECT * FROM search_history
            ORDER BY timestamp DESC
            LIMIT %s
            ''', (limit,))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
    
    def get_document_stats(self):
        """
        Get aggregate statistics about stored documents.
        
        Returns:
            Dictionary with document statistics
        """
        conn = self.get_connection()
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
                "document_types": {dt: count for dt, count in doc_types}
            }
        finally:
            cursor.close()
    
    def run_custom_query(self, query, params=None):
        """
        Run a custom SQL query on the database.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            Query results as a list of dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            if results:
                return [dict(row) for row in results]
            return []
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            cursor.close()

    def debug_tables(self):
        """Print information about database tables and content for debugging."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check tables in PostgreSQL
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = cursor.fetchall()
            print(f"Tables in database: {[t[0] for t in tables]}")
            
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
            cursor.close()