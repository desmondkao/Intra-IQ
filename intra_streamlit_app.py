import streamlit as st
import tempfile
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid
import sqlite3
import threading
from datetime import datetime
from dotenv import load_dotenv
from intraiq_standalone import IntraIQPipeline, DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL

# Fix for SQLite threading issues
# Thread-local storage for SQLite connections
thread_local = threading.local()

def get_db_connection(db_path):
    """Get a thread-local database connection."""
    if not hasattr(thread_local, "connection") or thread_local.connection is None:
        thread_local.connection = sqlite3.connect(db_path)
    return thread_local.connection

# ADD THIS FUNCTION TO FIX DOCUMENT PERSISTENCE
def fix_persistent_files_table(pipeline):
    """
    Fixes the document library by automatically adding existing documents to the persistent_files table.
    This ensures all documents appear in the library.
    
    Args:
        pipeline: The IntraIQPipeline instance with sql_store initialized
        
    Returns:
        int: Number of documents added to the persistent_files table
    """
    if not hasattr(pipeline, 'sql_store') or pipeline.sql_store is None:
        print("SQL store not available")
        return 0
    
    conn = sqlite3.connect(pipeline.sql_db_path)
    cursor = conn.cursor()
    
    # Check if binary_data column exists
    cursor.execute("PRAGMA table_info(documents)")
    columns = [col[1] for col in cursor.fetchall()]
    has_binary_data = 'binary_data' in columns
    
    # Find documents that exist in documents table but not in persistent_files
    if has_binary_data:
        # If binary_data exists, only add documents with binary data
        cursor.execute("""
        SELECT d.document_id, d.filename, d.file_size, d.upload_date, 
               d.binary_data IS NOT NULL as has_binary
        FROM documents d
        LEFT JOIN persistent_files p ON d.document_id = p.document_id
        WHERE p.document_id IS NULL AND d.binary_data IS NOT NULL
        """)
    else:
        # If binary_data doesn't exist, add all documents not in persistent_files
        cursor.execute("""
        SELECT d.document_id, d.filename, d.file_size, d.upload_date, 
               1 as has_binary
        FROM documents d
        LEFT JOIN persistent_files p ON d.document_id = p.document_id
        WHERE p.document_id IS NULL
        """)
    
    missing_docs = cursor.fetchall()
    added_count = 0
    
    for doc in missing_docs:
        document_id = doc[0]
        filename = doc[1]
        file_size = doc[2] or 0
        upload_date = doc[3] or datetime.now().isoformat()
        has_binary = doc[4]
        
        if has_binary:
            # Add to persistent_files table
            file_id = str(uuid.uuid4())
            cursor.execute("""
            INSERT INTO persistent_files 
            (file_id, document_id, filename, file_size, upload_date, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                file_id, 
                document_id, 
                filename, 
                file_size, 
                upload_date, 
                datetime.now().isoformat()
            ))
            added_count += 1
    
    conn.commit()
    conn.close()
    
    return added_count

# Enable foreign keys in SQLite
sqlite3.threadsafety = 0

# Load environment variables
load_dotenv()

# Page configuration with light theme
st.set_page_config(
    page_title="Hermes | Financial Document Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply light theme styling
st.markdown("""
<style>
    /* Light theme styling */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FFC107;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f7fb;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #2196F3;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-style: normal !important;
    }
    .source-header {
        font-weight: 600;
        color: #424242;
    }
    /* Fix content styling */
    .answer-box p em {
        font-style: normal !important;
    }
    .answer-box p strong {
        font-weight: 600;
        color: #1565C0;
    }
    /* Fix text area styling */
    textarea {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    /* Customize metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #1E88E5 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
    /* Button styling */
    div.stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1976D2;
        color: white;
    }
    /* Style source content */
    .source-content {
        font-family: monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    /* Make file uploader more visible */
    .css-14xtw13 {
        border: 2px dashed #1E88E5 !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
    }
    /* More aggressive fix for italic issues in the answer box */
    .answer-box em, 
    .answer-box p em, 
    .answer-box span em, 
    .answer-box * em, 
    em {
        font-style: normal !important;
    }
    
    /* Handle any text that might be using italic HTML tags */
    .answer-box i, 
    .answer-box p i, 
    .answer-box span i, 
    .answer-box * i, 
    i {
        font-style: normal !important;
    }
    
    /* Ensure all text in the answer box has normal font style */
    .answer-box, .answer-box * {
        font-style: normal !important;
    }
    
    /* Document card styling */
    .doc-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .doc-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #1E88E5;
    }
    
    .doc-card h4 {
        margin: 0;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "pipeline" not in st.session_state:
    # Create the pipeline with a consistent path for Qdrant
    qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
    os.makedirs(qdrant_path, exist_ok=True)
    
    st.session_state.pipeline = IntraIQPipeline(
        collection_name="hermes_documents",
        sql_db_path="hermes_documents.db",
        openai_model="gpt-3.5-turbo"  # Default model, will be updated when users change selection
    )
    
# Initialize uploaded_files in session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize document library tab
if "document_library_tab" not in st.session_state:
    st.session_state.document_library_tab = "Grid View"

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<div class="main-header">Hermes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Financial Document Analysis</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Settings section
    st.markdown("### Settings")
    
    # Check if embedding models are available
    embedding_models = [DEFAULT_EMBEDDING_MODEL]
    
    # Embedding model setting
    embedding_model = st.selectbox("Embedding Model", embedding_models, disabled=True)
    
    # Check if OpenAI is available
    try:
        from openai import OpenAI
        openai_available = True
    except ImportError:
        openai_available = False
    
    # LLM model selection
    llm_options = ["Local Model"]
    if openai_available:
        llm_options.append("ChatGPT")
        
    llm_option = st.selectbox(
        "Answer Generation Model",
        llm_options,
        index=0,
        help="Select which model to use for generating answers"
    )
    
    use_openai = (llm_option == "ChatGPT" and openai_available)
    
    if use_openai:
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Check if API key is available
        if not openai_api_key:
            st.warning("⚠️ No OpenAI API key found in environment variables. Please add it to your .env file.")
            st.markdown("""
            <div class="info-box">
            Add your OpenAI API key to a .env file in your project directory:
            <code>OPENAI_API_KEY=your-api-key-here</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ OpenAI API key loaded from environment variables")
        
        # Updated model selection with new options
        openai_model = st.selectbox(
            "OpenAI Model", 
            ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"], 
            index=0,
            help="Select which OpenAI model to use"
        )
        
        # Display estimated cost information with model-specific notes
        if openai_model == "gpt-3.5-turbo":
            st.info("💡 Using gpt-3.5-turbo (fastest, lowest cost)")
        elif openai_model == "gpt-4o":
            st.info("💡 Using gpt-4o (balanced performance/cost)")
        else:
            st.info("💡 Using gpt-4-turbo (most capable, highest cost)")
    else:
        # Check if transformers are available
        try:
            from transformers import AutoTokenizer
            transformers_available = True
            llm_models = [DEFAULT_LLM_MODEL]
        except ImportError:
            transformers_available = False
            llm_models = ["Not available - install transformers"]
            
        llm_model = st.selectbox("Local LLM Model", llm_models, disabled=not transformers_available)
        
        if not transformers_available:
            st.warning("⚠️ Transformers package not installed. Local models will not be available.")
    
    # Retrieval settings
    st.markdown("### Retrieval Settings")
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, 3)
    
    # Update pipeline settings
    st.session_state.pipeline.top_k = top_k
    
    # Agentic search section
    st.markdown("### Agentic Capabilities")
    
    # Check if needed packages are available for search agent
    try:
        import requests
        from openai import OpenAI
        search_agent_available = True
    except ImportError:
        search_agent_available = False
        
    enable_search_agent = st.checkbox(
        "Enable Current Market Information",
        value=False,
        help="When enabled, the system can search for current market information if your query requires it",
        disabled=not search_agent_available
    )
    
    if not search_agent_available and enable_search_agent:
        st.warning("⚠️ Search agent requires openai and requests packages. Please install them.")
    
    if enable_search_agent and search_agent_available:
        # Check for Tavily API key
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_api_key:
            st.warning("⚠️ No Tavily API key found. Please add it to your .env file.")
            st.markdown("""
            <div class="info-box">
            Add your Tavily API key to your .env file:
            <code>TAVILY_API_KEY=your-api-key-here</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ Tavily API key loaded from environment variables")
            
        # Also need OpenAI for the agent
        if not openai_api_key:
            st.warning("⚠️ The search agent also requires an OpenAI API key")

    # Document Library section
    st.divider()
    st.markdown("### 📚 Document Library")
    show_document_library = st.checkbox("Show Document Library", value=False)

    # Advanced features section
    st.divider()
    st.markdown("### Advanced Features")
    show_sql_features = st.checkbox("Show SQL Features", value=False)
    show_clustering = st.checkbox("Show Document Clustering", value=False)
    
    st.divider()
    
    # About section
    st.markdown("### About")
    st.markdown("""
    Hermes is a document analysis tool that uses Retrieval-Augmented Generation (RAG) to 
    provide accurate answers from your financial documents.
    
    Upload your financial documents, ask questions, and get insights instantly.
    """)

# --- Main Content ---
# Title
st.markdown('<div class="main-header">Hermes – Financial Document Analysis</div>', unsafe_allow_html=True)

# Upload Section
st.subheader("📄 Upload Documents")

st.markdown("""
<div class="info-box">
Upload your financial documents to analyze. Hermes supports PDF and TXT files.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload .pdf or .txt files", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

# Store uploaded files in session state for reference in other parts of the app
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        temp_dir = tempfile.mkdtemp()
        file_details = []
        
        # Add a checkbox to save files to the library
        save_to_library = st.checkbox("Save documents to library for future use", value=True)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                binary_data = uploaded_file.getbuffer()
                f.write(binary_data)
            file_details.append({
                "name": uploaded_file.name, 
                "size": f"{uploaded_file.size/1024:.1f} KB",
                "binary_data": binary_data if save_to_library else None
            })
        
        try:
            # Process documents with the pipeline - store document IDs 
            document_ids = st.session_state.pipeline.ingest_documents(temp_dir)
            
            # If requested, save binary data to SQL database
            if save_to_library and hasattr(st.session_state.pipeline, 'sql_store'):
                saved_count = 0
                for i, doc_id in enumerate(document_ids):
                    if doc_id and file_details[i]["binary_data"] is not None:
                        success = st.session_state.pipeline.sql_store.store_binary_document(
                            doc_id,
                            file_details[i]["name"],
                            file_details[i]["binary_data"]
                        )
                        if success:
                            saved_count += 1
                
                if saved_count > 0:
                    st.success(f"Saved {saved_count} document(s) to library for future use")
                
                # ADDED: Fix any documents not in persistent_files table
                fixed_count = fix_persistent_files_table(st.session_state.pipeline)
                if fixed_count > 0:
                    st.success(f"✨ Fixed {fixed_count} document(s) in library")
            
            # Show success message and document details
            st.markdown(f"""
            <div class="success-box">
            Successfully processed {len(uploaded_files)} document(s)!
            {" Documents saved to library for future use." if save_to_library else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Show document details
            for doc in file_details:
                st.markdown(f"📄 **{doc['name']}** ({doc['size']})")
            
            # Show brief document stats
            try:
                stats = {
                    "document_count": 0,
                    "chunk_count": 0,
                    "total_word_count": 0
                }
                
                # Use thread-safe connection
                conn = get_db_connection(st.session_state.pipeline.sql_db_path)
                cursor = conn.cursor()
                
                try:
                    cursor.execute('SELECT COUNT(*) FROM documents')
                    stats["document_count"] = cursor.fetchone()[0] or 0
                    cursor.execute('SELECT COUNT(*) FROM chunks')
                    stats["chunk_count"] = cursor.fetchone()[0] or 0
                    cursor.execute('SELECT SUM(word_count) FROM chunks')
                    stats["total_word_count"] = cursor.fetchone()[0] or 0
                except Exception as e:
                    # Silently ignore errors, as these are optional metrics
                    pass
                
                # Display simple stats
                st.markdown("### 📊 Document Stats")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats["document_count"])
                with col2:
                    st.metric("Chunks", stats["chunk_count"])
                with col3:
                    st.metric("Total Words", f"{int(stats['total_word_count']):,}")
            except Exception as e:
                # Silently handle errors with document stats
                pass
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    # Document clustering if enabled
    if show_clustering:
        st.markdown("### 🔍 Document Structure Analysis")
        if st.button("Analyze Document Structure with K-means"):
            with st.spinner("Applying K-means clustering to identify document topics..."):
                # Get all embeddings and chunks
                all_embeddings = []
                all_chunks = []
                
                try:
                    # Check if required packages are available
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.decomposition import PCA
                        sklearn_available = True
                    except ImportError:
                        sklearn_available = False
                        st.error("scikit-learn is required for clustering. Please install it with: pip install scikit-learn")
                    
                    if sklearn_available:
                        # In a real scenario, you would extract these from your RAG pipeline
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            document = st.session_state.pipeline.processor.process_document(file_path)
                            embedded_doc = st.session_state.pipeline.embedder.embed_document(document)
                            
                            all_embeddings.extend(embedded_doc["embeddings"])
                            all_chunks.extend(document["chunks"])
                        
                        # Apply K-means clustering
                        n_clusters = min(5, len(all_chunks))  # Don't use more clusters than chunks
                        
                        if n_clusters < 2 or len(all_chunks) < 2:
                            st.warning("Not enough document chunks for meaningful clustering. Please upload more content.")
                        else:
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(all_embeddings)
                            
                            # Reduce dimensions for visualization
                            pca = PCA(n_components=2)
                            reduced_embeddings = pca.fit_transform(all_embeddings)
                            
                            # Create visualization with light theme
                            plt.style.use('default')
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Plot points with cluster colors
                            scatter = ax.scatter(
                                reduced_embeddings[:, 0], 
                                reduced_embeddings[:, 1],
                                c=clusters, 
                                cmap='viridis', 
                                alpha=0.7,
                                s=80
                            )
                            
                            # Add cluster centers
                            centers = kmeans.cluster_centers_
                            reduced_centers = pca.transform(centers)
                            ax.scatter(
                                reduced_centers[:, 0], 
                                reduced_centers[:, 1], 
                                c='red',
                                marker='X', 
                                s=200, 
                                alpha=1,
                                label='Cluster Centers'
                            )
                            
                            # Add legend and labels
                            ax.legend()
                            ax.set_title(f'K-means Clustering of Document Content (k={n_clusters})')
                            ax.set_xlabel('Principal Component 1')
                            ax.set_ylabel('Principal Component 2')
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Display results
                            st.pyplot(fig)
                            
                            # Extract cluster characteristics
                            st.markdown("### 🔍 Document Topics")
                            
                            for i in range(n_clusters):
                                # Get chunks in this cluster
                                cluster_chunks = [all_chunks[j] for j, label in enumerate(clusters) if label == i]
                                
                                if cluster_chunks:
                                    # Get all text in this cluster
                                    cluster_text = " ".join(cluster_chunks)
                                    
                                    # Simple word frequency for demonstration
                                    words = cluster_text.lower().split()
                                    word_freq = {}
                                    stopwords = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'that', 'with', 'as', 'on'}
                                    
                                    for word in words:
                                        if word not in stopwords and len(word) > 3:
                                            word_freq[word] = word_freq.get(word, 0) + 1
                                    
                                    # Get top words
                                    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                                    
                                    # Display cluster information
                                    st.subheader(f"Topic {i+1}")
                                    
                                    # Show top words as bar chart
                                    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                                    
                                    # Create visualization with light theme
                                    plt.style.use('default')
                                    fig, ax = plt.subplots(figsize=(10, 3))
                                    
                                    bars = ax.bar(top_words_df['Word'], top_words_df['Frequency'], color='#1E88E5')
                                    
                                    # Add value labels to bars
                                    for bar in bars:
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                f'{height:.0f}',
                                                ha='center', va='bottom')
                                    
                                    ax.set_title(f"Key Terms in Topic {i+1}")
                                    ax.set_ylabel('Frequency')
                                    ax.set_ylim(0, max(top_words_df['Frequency']) * 1.2)  # Add some headroom for labels
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Show sample text from this cluster
                                    with st.expander(f"Sample content from Topic {i+1}"):
                                        st.markdown(f"<div class='source-content'>{cluster_chunks[0][:500] + '...' if len(cluster_chunks[0]) > 500 else cluster_chunks[0]}</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in clustering analysis: {str(e)}")

# Document Library Section
if show_document_library:
    st.markdown("---")
    st.header("📚 Document Library")
    
    # ADDED: Fix documents button
    if st.button("🔧 Fix Document Library"):
        if hasattr(st.session_state.pipeline, 'sql_store'):
            with st.spinner("Fixing document library..."):
                fixed_count = fix_persistent_files_table(st.session_state.pipeline)
                if fixed_count > 0:
                    st.success(f"Fixed {fixed_count} document(s) in the library")
                    st.rerun()  # Refresh the page
                else:
                    st.info("No documents needed to be fixed")
    
    # Check if SQL store is available
    if hasattr(st.session_state.pipeline, 'sql_store'):
        # Get list of stored documents
        stored_docs = st.session_state.pipeline.sql_store.get_all_stored_documents()
        
        if stored_docs:
            st.success(f"Found {len(stored_docs)} saved documents in your library")
            
            # Create tabs for different views
            tab_options = ["Grid View", "Table View", "Analytics"]
            st.session_state.document_library_tab = st.radio(
                "View", tab_options, 
                horizontal=True, 
                index=tab_options.index(st.session_state.document_library_tab)
            )
            
            if st.session_state.document_library_tab == "Grid View":
                # Grid view with cards
                cols = st.columns(3)
                for i, doc in enumerate(stored_docs):
                    with cols[i % 3]:
                        with st.container():
                            st.markdown(f"""
                            <div class="doc-card">
                                <h4>{doc['filename']}</h4>
                                <p><small>Added: {doc['upload_date'][:10]}</small></p>
                                <p>Type: {doc['document_type'] or 'Unknown'}<br>
                                Size: {doc['file_size']/1024:.1f} KB<br>
                                Views: {doc['access_count']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Load {i}", key=f"load_{doc['document_id']}"):
                                    # Get binary data
                                    binary_doc = st.session_state.pipeline.sql_store.get_binary_document(doc['document_id'])
                                    if binary_doc and binary_doc['binary_data']:
                                        # Create a temp file and process it
                                        with st.spinner(f"Loading {doc['filename']}..."):
                                            temp_dir = tempfile.mkdtemp()
                                            file_path = os.path.join(temp_dir, doc['filename'])
                                            with open(file_path, "wb") as f:
                                                f.write(binary_doc['binary_data'])
                                            
                                            # Now process the document with the pipeline
                                            st.session_state.pipeline.ingest_documents(temp_dir)
                                            
                                            # Show success message
                                            st.success(f"📄 Loaded {doc['filename']} - ready to query!")
                                    else:
                                        st.error(f"Could not retrieve data for {doc['filename']}")
                            
                            with col2:
                                if st.button(f"Delete {i}", key=f"delete_{doc['document_id']}"):
                                    # Confirm deletion
                                    if st.session_state.pipeline.sql_store.delete_stored_document(doc['document_id']):
                                        st.success(f"Removed {doc['filename']} from library")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete document")
            
            elif st.session_state.document_library_tab == "Table View":
                # Table view
                table_data = []
                for doc in stored_docs:
                    table_data.append({
                        "Filename": doc['filename'],
                        "Type": doc['document_type'] or "Unknown",
                        "Size (KB)": f"{doc['file_size']/1024:.1f}",
                        "Upload Date": doc['upload_date'][:10],
                        "Views": doc['access_count']
                    })
                
                # Convert to DataFrame for display
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # Document selection for actions
                selected_docs = st.multiselect(
                    "Select documents for bulk actions:",
                    [doc['filename'] for doc in stored_docs]
                )
                
                # Bulk actions row
                if selected_docs:
                    st.subheader("Document Actions")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load Selected Documents", key="bulk_load"):
                            loaded_count = 0
                            with st.spinner("Loading selected documents..."):
                                for doc in stored_docs:
                                    if doc['filename'] in selected_docs:
                                        binary_doc = st.session_state.pipeline.sql_store.get_binary_document(doc['document_id'])
                                        if binary_doc and binary_doc['binary_data']:
                                            temp_dir = tempfile.mkdtemp()
                                            file_path = os.path.join(temp_dir, doc['filename'])
                                            with open(file_path, "wb") as f:
                                                f.write(binary_doc['binary_data'])
                                            
                                            # Process with pipeline
                                            st.session_state.pipeline.ingest_documents(temp_dir)
                                            loaded_count += 1
                                            
                            st.success(f"Loaded {loaded_count} document(s) - ready to query!")
                    
                    with col2:
                        if st.button("Remove Selected Documents", key="bulk_delete"):
                            deleted_count = 0
                            with st.spinner("Removing selected documents..."):
                                for doc in stored_docs:
                                    if doc['filename'] in selected_docs:
                                        if st.session_state.pipeline.sql_store.delete_stored_document(doc['document_id']):
                                            deleted_count += 1
                                            
                            st.success(f"Removed {deleted_count} document(s) from library")
                            st.rerun()
            
            else:  # Analytics tab
                # Display library statistics
                st.subheader("Document Library Analytics")
                
                # Get detailed stats
                stats = st.session_state.pipeline.sql_store.get_document_stats()
                
                # Basic metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", stats.get("persistent_document_count", 0))
                with col2:
                    st.metric("Total Size", f"{sum(doc['file_size'] for doc in stored_docs)/1024:.1f} KB")
                with col3:
                    st.metric("Total Views", sum(doc['access_count'] for doc in stored_docs))
                
                # Document type breakdown
                st.subheader("Document Types")
                doc_types = {}
                for doc in stored_docs:
                    doc_type = doc['document_type'] or "Unknown"
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Create pie chart
                if doc_types:
                    plt.style.use('default')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    labels = list(doc_types.keys())
                    sizes = list(doc_types.values())
                    
                    # Custom colors for light theme
                    colors = ['#1E88E5', '#4CAF50', '#FFC107', '#FF5252', '#9C27B0']
                    
                    # Create pie chart with percentage labels
                    wedges, texts, autotexts = ax.pie(
                        sizes, 
                        labels=labels, 
                        autopct='%1.1f%%',
                        colors=colors[:len(doc_types)],
                        startangle=90,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                    )
                    
                    # Style the percentage text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('Document Types in Library')
                    ax.axis('equal')  # Equal aspect ratio ensures pie is circular
                    
                    st.pyplot(fig)
                
                # Document usage over time
                st.subheader("Document Usage")
                
                # Get access data by document
                access_data = {}
                for doc in stored_docs:
                    doc_name = doc['filename']
                    if len(doc_name) > 25:  # Truncate long filenames
                        doc_name = doc_name[:22] + '...'
                    access_data[doc_name] = doc['access_count']
                
                # Create bar chart
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort by access count
                sorted_data = dict(sorted(access_data.items(), key=lambda x: x[1], reverse=True))
                
                bars = ax.bar(
                    sorted_data.keys(), 
                    sorted_data.values(),
                    color='#1E88E5'
                )
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width()/2., 
                            height + 0.1,
                            f'{height:.0f}',
                            ha='center', 
                            va='bottom'
                        )
                
                ax.set_title('Document Usage (Number of Accesses)')
                ax.set_ylabel('Access Count')
                ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Most recently accessed documents
                st.subheader("Recently Accessed Documents")
                
                # Sort by last accessed time
                recent_docs = sorted(
                    stored_docs, 
                    key=lambda x: x.get('upload_date', ''), 
                    reverse=True
                )[:5]
                
                if recent_docs:
                    recent_data = []
                    for doc in recent_docs:
                        recent_data.append({
                            "Filename": doc['filename'],
                            "Last Accessed": doc.get('upload_date', 'Unknown')[:16].replace('T', ' '),
                            "Access Count": doc['access_count']
                        })
                    
                    st.dataframe(pd.DataFrame(recent_data), use_container_width=True)
                else:
                    st.info("No recently accessed documents")
                
                # Storage management
                st.subheader("Storage Management")
                
                # Calculate total storage used
                total_size_kb = sum(doc['file_size'] for doc in stored_docs) / 1024
                
                # Show available storage (for demo purposes)
                storage_limit_kb = 100 * 1024  # 100MB example limit
                storage_used_pct = (total_size_kb / storage_limit_kb) * 100
                
                # Create a progress bar
                st.markdown(f"**Storage Used**: {total_size_kb:.1f} KB of {storage_limit_kb/1024:.1f} MB ({storage_used_pct:.1f}%)")
                st.progress(min(storage_used_pct/100, 1.0))
                
                if storage_used_pct > 80:
                    st.warning("Storage usage is high. Consider removing unused documents.")
                
                # ADDED: Fix any unfixed documents automatically
                conn = sqlite3.connect(st.session_state.pipeline.sql_db_path)
                cursor = conn.cursor()
                
                # Check if binary_data column exists
                cursor.execute("PRAGMA table_info(documents)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'binary_data' in columns:
                    cursor.execute("""
                    SELECT COUNT(*) FROM documents d
                    LEFT JOIN persistent_files p ON d.document_id = p.document_id
                    WHERE p.document_id IS NULL AND d.binary_data IS NOT NULL
                    """)
                else:
                    cursor.execute("""
                    SELECT COUNT(*) FROM documents d
                    LEFT JOIN persistent_files p ON d.document_id = p.document_id
                    WHERE p.document_id IS NULL
                    """)
                    
                unfixed_count = cursor.fetchone()[0]
                conn.close()
                
                if unfixed_count > 0:
                    st.warning(f"Found {unfixed_count} document(s) not properly registered in the library")
                    if st.button("Fix Missing Documents"):
                        fixed = fix_persistent_files_table(st.session_state.pipeline)
                        if fixed > 0:
                            st.success(f"Fixed {fixed} document(s)")
                            st.rerun()
                
                # Add a cleanup button for unused documents
                if st.button("Clean Unused Documents"):
                    # Find documents with no access
                    unused_docs = [doc for doc in stored_docs if doc['access_count'] <= 1]
                    
                    if unused_docs:
                        deleted = 0
                        with st.spinner(f"Cleaning {len(unused_docs)} unused documents..."):
                            for doc in unused_docs:
                                if st.session_state.pipeline.sql_store.delete_stored_document(doc['document_id']):
                                    deleted += 1
                        
                        if deleted > 0:
                            st.success(f"Removed {deleted} unused documents, freeing up space")
                            st.rerun()
                        else:
                            st.error("Failed to remove unused documents")
                    else:
                        st.info("No unused documents found to clean up")
        else:
            st.info("Your document library is empty. Upload documents and enable 'Save documents to library' to build your library.")
            
            # Check if there are documents that need fixing
            conn = sqlite3.connect(st.session_state.pipeline.sql_db_path)
            cursor = conn.cursor()
            
            # Check if binary_data column exists
            cursor.execute("PRAGMA table_info(documents)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'binary_data' in columns:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE binary_data IS NOT NULL")
            else:
                cursor.execute("SELECT COUNT(*) FROM documents")
                
            unfixed_docs = cursor.fetchone()[0]
            conn.close()
            
            if unfixed_docs > 0:
                st.warning(f"Found {unfixed_docs} document(s) that need to be added to the library")
                if st.button("Fix Documents"):
                    fixed_count = fix_persistent_files_table(st.session_state.pipeline)
                    if fixed_count > 0:
                        st.success(f"Added {fixed_count} document(s) to your library")
                        st.rerun()  # Refresh the page to show the documents
            
            # Add explanation about persistence
            st.markdown("""
            <div class="info-box">
            <h4>About Document Persistence</h4>
            <p>The document library allows you to store your financial documents for future analysis sessions. When you save a document to the library, it will be stored in the SQLite database for future use, enabling you to:</p>
            <ul>
                <li>Access your documents across different Hermes sessions</li>
                <li>Avoid re-uploading frequently used documents</li>
                <li>Track document usage statistics</li>
                <li>Organize your financial documents in one place</li>
            </ul>
            <p>To save documents, simply upload them with the "Save documents to library" option enabled.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample upload with save option highlighted
            st.markdown("""
            <div style="border:1px solid #ddd; border-radius:5px; padding:15px; margin-top:20px;">
                <h4 style="margin-top:0">How to Save Documents</h4>
                <ol>
                    <li>Upload your financial documents using the file uploader above</li>
                    <li>Make sure the "Save documents to library for future use" checkbox is selected</li>
                    <li>Your documents will be processed and saved to the library</li>
                    <li>Return to the Document Library anytime to access your saved documents</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Document library requires SQL support. Please ensure that SQLite is properly configured.")

# SQL Features Section (Advanced)
if show_sql_features and (uploaded_files or (show_document_library and hasattr(st.session_state.pipeline, 'sql_store') and st.session_state.pipeline.sql_store.get_all_stored_documents())):
    st.markdown("---")
    st.header("📊 SQL Database Features")
    st.markdown("This section demonstrates SQL capabilities for document management and search.")
    
    # Only show if SQL store is available
    if hasattr(st.session_state.pipeline, 'sql_store'):
        # Create a simplified SQL search interface
        st.subheader("SQL Keyword Search")
        sql_search_term = st.text_input(
            "Search documents with SQL LIKE query:", 
            placeholder="e.g., revenue OR profit OR financial"
        )
        
        if sql_search_term:
            try:
                # Connect directly to avoid threading issues
                conn = get_db_connection(st.session_state.pipeline.sql_db_path)
                cursor = conn.cursor()
                
                # Use SQL LIKE for basic text search
                search_pattern = f"%{sql_search_term}%"
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
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} matches")
                    
                    for i, row in enumerate(results):
                        result = dict(zip(columns, row))
                        with st.expander(f"Match {i+1}: {result['filename']} (Chunk {result['chunk_index']})"):
                            st.markdown(f"<div class='source-header'>Word count: {result['word_count']}</div>", unsafe_allow_html=True)
                            
                            st.markdown("<div class='source-header'>Content:</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='source-content'>{result['text']}</div>", unsafe_allow_html=True)
                else:
                    st.info("No matches found")
                
            except Exception as e:
                st.error(f"SQL Search error: {str(e)}")
        
        # Show a simple custom SQL query interface
        with st.expander("Run Custom SQL Query"):
            st.markdown("""
            This feature demonstrates custom SQL analytics on the document database. 
            SQL allows for flexible querying and analysis of document metadata and content.
            """)
            
            # Example queries dropdown
            example_queries = {
                "Select a query": "",
                "Document Summary": "SELECT document_id, filename FROM documents",
                "Top 5 Largest Chunks": "SELECT document_id, chunk_index, word_count FROM chunks ORDER BY word_count DESC LIMIT 5",
                "Search for financial terms": "SELECT document_id, chunk_index, text FROM chunks WHERE text LIKE '%financial%' LIMIT 3",
                "Get document date estimates": "SELECT query, timestamp, document_date_estimate FROM search_history WHERE document_date_estimate IS NOT NULL AND document_date_estimate != 'Unknown' ORDER BY timestamp DESC LIMIT 5",
                "Library document stats": "SELECT p.document_id, p.filename, p.access_count, p.upload_date FROM persistent_files p ORDER BY p.access_count DESC"
            }
            
            selected_example = st.selectbox("Example Queries:", options=list(example_queries.keys()))
            
            # SQL query input
            custom_query = st.text_area(
                "Enter SQL Query (be careful with complex queries):",
                value=example_queries[selected_example] if selected_example in example_queries else "",
                height=100
            )
            
            if custom_query and st.button("Run Query"):
                try:
                    # Connect directly to avoid threading issues
                    conn = get_db_connection(st.session_state.pipeline.sql_db_path)
                    cursor = conn.cursor()
                    
                    # Execute query
                    cursor.execute(custom_query)
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Convert to DataFrame for display
                    if results:
                        results_df = pd.DataFrame(results, columns=columns)
                        st.dataframe(results_df)
                        st.caption(f"Query returned {len(results)} rows")
                    else:
                        st.info("Query returned no results")
                    
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        
        # Document Analytics
        try:
            conn = get_db_connection(st.session_state.pipeline.sql_db_path)
            cursor = conn.cursor()
            
            # Get document type distribution
            cursor.execute('''
            SELECT document_type, COUNT(*) as count
            FROM documents
            GROUP BY document_type
            ''')
            doc_types = cursor.fetchall()
            
            if doc_types:
                st.subheader("Document Type Distribution")
                
                # Create pie chart with light theme
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(10, 6))
                
                labels = [t[0] for t in doc_types]
                sizes = [t[1] for t in doc_types]
                
                # Custom colors for light theme
                colors = ['#1E88E5', '#4CAF50', '#FFC107', '#FF5252', '#9C27B0']
                
                # Create pie chart with percentage labels
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels, 
                    autopct='%1.1f%%',
                    colors=colors[:len(doc_types)],
                    startangle=90,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                )
                
                # Style the percentage text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Document Types in Database')
                ax.axis('equal')  # Equal aspect ratio ensures pie is circular
                
                st.pyplot(fig)
            
            # Add a new section to visualize model usage stats
            st.subheader("Model Usage Statistics")
            try:
                # Check if the model_used column exists
                cursor.execute("PRAGMA table_info(search_history)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'model_used' in columns:
                    cursor.execute('''
                    SELECT model_used, COUNT(*) as count
                    FROM search_history
                    WHERE model_used IS NOT NULL
                    GROUP BY model_used
                    ''')
                    model_stats = cursor.fetchall()
                    
                    if model_stats:
                        # Create bar chart for model usage
                        plt.style.use('default')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        models = [stat[0] if stat[0] else "Not specified" for stat in model_stats]
                        counts = [stat[1] for stat in model_stats]
                        
                        # Bar chart
                        bars = ax.bar(models, counts, color='#1E88E5')
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.0f}',
                                    ha='center', va='bottom')
                        
                        ax.set_title('OpenAI Model Usage')
                        ax.set_ylabel('Number of Queries')
                        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.info("No model usage data available yet")
            except Exception as e:
                # Just skip this section if there's an error
                pass
                
        except Exception as e:
            # Silently ignore errors in analytics
            pass
            
    else:
        st.warning("SQL features are not available. Make sure the SQL components are properly installed.")

# QA Section - Primary Feature
st.markdown("---")
st.subheader("❓ Ask a Question About Your Documents")

# ADDED: Check if there are documents that need fixing before allowing questions
if hasattr(st.session_state.pipeline, 'sql_store'):
    conn = sqlite3.connect(st.session_state.pipeline.sql_db_path)
    cursor = conn.cursor()
    
    # First check if binary_data column exists
    cursor.execute("PRAGMA table_info(documents)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'binary_data' in columns:
        # Check for documents not in persistent_files
        cursor.execute("""
        SELECT COUNT(*) FROM documents d
        LEFT JOIN persistent_files p ON d.document_id = p.document_id
        WHERE p.document_id IS NULL AND d.binary_data IS NOT NULL
        """)
        unfixed_count = cursor.fetchone()[0]
        
        if unfixed_count > 0:
            st.warning(f"⚠️ Found {unfixed_count} document(s) not properly registered in the library. This may affect search results.")
            if st.button("Fix Documents Now"):
                fixed = fix_persistent_files_table(st.session_state.pipeline)
                if fixed > 0:
                    st.success(f"✅ Fixed {fixed} document(s) in the library")
                    st.rerun()
    else:
        # If binary_data column doesn't exist, use a simpler check
        cursor.execute("""
        SELECT COUNT(*) FROM documents d
        LEFT JOIN persistent_files p ON d.document_id = p.document_id
        WHERE p.document_id IS NULL
        """)
        unfixed_count = cursor.fetchone()[0]
        
        if unfixed_count > 0:
            st.warning(f"⚠️ Found {unfixed_count} document(s) not properly registered in the library. This may affect search results.")
            if st.button("Fix Documents Now"):
                fixed = fix_persistent_files_table(st.session_state.pipeline)
                if fixed > 0:
                    st.success(f"✅ Fixed {fixed} document(s) in the library")
                    st.rerun()
    
    conn.close()

if not uploaded_files and not (show_document_library and hasattr(st.session_state.pipeline, 'sql_store') and st.session_state.pipeline.sql_store.get_all_stored_documents()):
    # Check if there might be documents that just need fixing
    if hasattr(st.session_state.pipeline, 'sql_store'):
        conn = sqlite3.connect(st.session_state.pipeline.sql_db_path)
        cursor = conn.cursor()
        
        # First check if binary_data column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'binary_data' in columns:
            cursor.execute("SELECT COUNT(*) FROM documents WHERE binary_data IS NOT NULL")
        else:
            # If binary_data column doesn't exist, just count all documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            
        unfixed_docs = cursor.fetchone()[0]
        conn.close()
        
        if unfixed_docs > 0:
            st.warning(f"⚠️ Found {unfixed_docs} document(s) that aren't visible in your library. Click 'Fix Documents' to make them available.")
            if st.button("Fix Documents", key="fix_docs_question"):
                fixed_count = fix_persistent_files_table(st.session_state.pipeline)
                if fixed_count > 0:
                    st.success(f"✅ Fixed {fixed_count} document(s). They're now available in your library!")
                    st.rerun()
    
    st.markdown("""
    <div class="warning-box">
    Please upload at least one document or load a document from your library to start asking questions.
    </div>
    """, unsafe_allow_html=True)
else:
    query = st.text_input("Type your question here:", placeholder="e.g., What are the main findings of this report?")
    
    if query:
        with st.spinner("Analyzing documents and generating answer..."):
            # Determine which LLM to use
            use_openai_for_query = False
            openai_api_key_for_query = None
            openai_model_for_query = None
            
            if use_openai:
                openai_api_key_for_query = os.getenv("OPENAI_API_KEY", "")
                if not openai_api_key_for_query:
                    st.warning("Please provide an OpenAI API key to use OpenAI for answering.")
                else:
                    use_openai_for_query = True
                    openai_model_for_query = openai_model
                    
                    # Update the pipeline's default model as well
                    st.session_state.pipeline.openai_model = openai_model
            
            try:
                # Run the RAG pipeline
                result = st.session_state.pipeline.run_rag(
                    query, 
                    use_openai=use_openai_for_query,
                    openai_api_key=openai_api_key_for_query,
                    openai_model=openai_model_for_query,
                    enable_search_agent=enable_search_agent
                )
                
                # Display the answer
                st.markdown("### 💬 Answer")
                
                # Show search information if available
                if "search_performed" in result and result["search_performed"]:
                    if result.get("search_successful", False):
                        st.success(f"✅ Enhanced with current market information")
                        
                        # Display document date estimate if available
                        if "document_date_estimate" in result and result["document_date_estimate"] and result["document_date_estimate"].lower() != "unknown":
                            st.info(f"📅 Document information from approximately {result['document_date_estimate']}")
                            
                        st.markdown(f"*Search query: \"{result.get('search_query', query)}\"*")
                    else:
                        st.warning(f"ℹ️ Attempted to find current information but encountered an issue: {result.get('search_error', 'Unknown error')}")
                
                # Display the answer in a styled box
                st.markdown(f"""
                <div class="answer-box">
                {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if result.get("sources"):
                    st.markdown("### 📚 Sources")
                    
                    # Group sources by type
                    document_sources = [src for src in result["sources"] if src.get("source_type", "") != "web_search"]
                    web_sources = [src for src in result["sources"] if src.get("source_type", "") == "web_search"]
                    
                    # Display unified source container
                    with st.container():
                        # Display document sources
                        if document_sources:
                            st.markdown("#### Document Sources")
                            for i, src in enumerate(document_sources, 1):
                                with st.expander(f"Source {i} – {src['document_name']} (Relevance: {src['score']:.4f})"):
                                    st.markdown(f"<div class='source-header'>Document: {src['document_name']}</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='source-header'>Relevance Score: {src['score']:.4f}</div>", unsafe_allow_html=True)
                                    st.markdown("**Content:**")
                                    
                                    # Use styled content display
                                    st.markdown(f"""
                                    <div class="source-content">
                                    {src["text"].replace('<', '&lt;').replace('>', '&gt;')}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Display web sources
                        if web_sources:
                            st.markdown("#### Current Market Information Sources")
                            for i, src in enumerate(web_sources, 1):
                                with st.expander(f"Web Source {i} – {src['document_name']}"):
                                    if "url" in src:
                                        st.markdown(f"**Source:** [{src['document_name']}]({src['url']})")
                                    
                                    # Only show published date if it's known and not "Unknown"
                                    if "published_date" in src and src["published_date"] and src["published_date"].lower() != "unknown":
                                        st.markdown(f"**Published:** {src['published_date']}")
                                    
                                    st.markdown("**Content:**")
                                    st.markdown(f"<div class='source-content'>{src['text'][:1000] + '...' if len(src['text']) > 1000 else src['text']}</div>", unsafe_allow_html=True)
                    
                    # Show similarity visualization
                    if document_sources:
                        st.markdown("### 📊 Query-Document Similarity Analysis")
                        
                        # Extract document texts and scores
                        docs = [src["document_name"] for src in document_sources]
                        scores = [src["score"] for src in document_sources]
                        
                        # Create bar chart with light theme
                        plt.style.use('default')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        y_pos = range(len(docs))
                        bars = ax.barh(y_pos, scores, color='#1E88E5')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(docs)
                        ax.invert_yaxis()  # Labels read top-to-bottom
                        ax.set_xlabel('Similarity Score')
                        ax.set_title('Query-Document Relevance')
                        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
                        
                        # Add value labels
                        for i, v in enumerate(scores):
                            ax.text(v + 0.01, i, f"{v:.4f}", va='center')
                        
                        st.pyplot(fig)
                    
                    # Export options
                    st.markdown("### Export Options")
                    export_format = st.selectbox("Export format:", ["None", "Text", "JSON"])
                    
                    if export_format != "None" and st.button("Export Results"):
                        if export_format == "Text":
                            export_text = f"""
                            Question: {query}
                            
                            Answer: {result["answer"]}
                            
                            Sources:
                            """
                            # Add document sources
                            for i, src in enumerate(document_sources, 1):
                                export_text += f"\nSource {i}: {src['document_name']} (Score: {src['score']:.4f})\n"
                                export_text += f"{src['text'][:500]}...\n"
                                
                            # Add web sources if available
                            if web_sources:
                                export_text += "\nCurrent Market Information:\n"
                                for i, src in enumerate(web_sources, 1):
                                    export_text += f"\nWeb Source {i}: {src['document_name']}\n"
                                    if "url" in src:
                                        export_text += f"URL: {src['url']}\n"
                                    if "published_date" in src:
                                        export_text += f"Published: {src['published_date']}\n"
                                    export_text += f"{src['text'][:500]}...\n"
                            
                            st.download_button(
                                label="Download Text",
                                data=export_text,
                                file_name="hermes_results.txt",
                                mime="text/plain"
                            )
                        elif export_format == "JSON":
                            export_data = {
                                "question": query,
                                "answer": result["answer"],
                                "sources": result["sources"]
                            }
                            
                            # Add search information if available
                            if "search_performed" in result:
                                export_data["search_info"] = {
                                    "search_performed": result["search_performed"],
                                    "search_successful": result.get("search_successful", False),
                                    "search_query": result.get("search_query", ""),
                                    "reasoning": result.get("reasoning", ""),
                                    "document_date_estimate": result.get("document_date_estimate", "Unknown")
                                }
                            
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name="hermes_results.json",
                                mime="application/json"
                            )
                
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 20px 0;">
Hermes - Financial Document Analysis System | RAG-powered insights
</div>
""", unsafe_allow_html=True)