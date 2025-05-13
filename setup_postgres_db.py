# PostgreSQL Database Setup Script
# Run this before starting your Streamlit app to create the database

import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection parameters from environment
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "hermes_documents")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")

def setup_database():
    """Create the PostgreSQL database and required tables."""
    print(f"Setting up PostgreSQL database: {DB_NAME}")
    
    # First connect to default 'postgres' database to create our database
    try:
        # Connect to default database
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT,
            user=DB_USER, 
            password=DB_PASSWORD,
            dbname="postgres"
        )
        conn.autocommit = True  # Needed for creating database
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{DB_NAME}'...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to PostgreSQL or creating database: {str(e)}")
        return False
    
    # Now connect to our new database and create tables
    try:
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT,
            user=DB_USER, 
            password=DB_PASSWORD,
            dbname=DB_NAME
        )
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            document_type TEXT NOT NULL,
            binary_data BYTEA,
            text_content TEXT,
            embedding_model TEXT,
            last_accessed TEXT
        )
        ''')
        
        # Create chunks table
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
        
        # Create search_history table
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
        
        # Create persistent_files table
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
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_docs_filename ON documents(filename)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_persistent_files_document_id ON persistent_files(document_id)')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Tables created successfully.")
        return True
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_database():
        print("PostgreSQL database setup completed successfully!")
    else:
        print("Failed to set up PostgreSQL database. Check connection parameters and try again.")