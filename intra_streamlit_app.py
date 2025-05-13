import streamlit as st
import tempfile
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid
import threading
from datetime import datetime
from dotenv import load_dotenv
from intraiq_standalone import IntraIQPipeline, DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL

# Thread-local storage
thread_local = threading.local()

# Load environment variables
load_dotenv()

# Page configuration with light theme
st.set_page_config(
    page_title="Opus | Financial Document Analysis",
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
    # Create the pipeline
    qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_storage")
    os.makedirs(qdrant_path, exist_ok=True)
    
    st.session_state.pipeline = IntraIQPipeline(
        collection_name="opus_documents",
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
    st.markdown('<div class="main-header">Opus</div>', unsafe_allow_html=True)
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
            st.warning("No OpenAI API key found in environment variables. Please add it to your .env file.")
            st.markdown("""
            <div class="info-box">
            Add your OpenAI API key to a .env file in your project directory:
            <code>OPENAI_API_KEY=your-api-key-here</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("OpenAI API key loaded from environment variables")
        
        # Updated model selection with new options
        openai_model = st.selectbox(
            "OpenAI Model", 
            ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"], 
            index=0,
            help="Select which OpenAI model to use"
        )
        
        # Display estimated cost information with model-specific notes
        if openai_model == "gpt-3.5-turbo":
            st.info("Using gpt-3.5-turbo (fastest, lowest cost)")
        elif openai_model == "gpt-4o":
            st.info("Using gpt-4o (balanced performance/cost)")
        else:
            st.info("Using gpt-4-turbo (most capable, highest cost)")
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
            st.warning("Transformers package not installed. Local models will not be available.")
    
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
        st.warning("Search agent requires openai and requests packages. Please install them.")
    
    if enable_search_agent and search_agent_available:
        # Check for Tavily API key
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_api_key:
            st.warning("No Tavily API key found. Please add it to your .env file.")
            st.markdown("""
            <div class="info-box">
            Add your Tavily API key to your .env file:
            <code>TAVILY_API_KEY=your-api-key-here</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("Tavily API key loaded from environment variables")
            
        # Also need OpenAI for the agent
        if not openai_api_key:
            st.warning("The search agent also requires an OpenAI API key")

    # Document Library section
    st.divider()
    st.markdown("### Document Library")
    show_document_library = st.checkbox("Show Document Library", value=False)

    # Advanced features section
    st.divider()
    st.markdown("### Advanced Features")
    show_clustering = st.checkbox("Show Document Clustering", value=False)
    
    # About section
    st.divider()
    st.markdown("### About")
    st.markdown("""
    Opus is a document analysis tool that uses Retrieval-Augmented Generation (RAG) to 
    provide accurate answers from your financial documents.
    
    Upload your financial documents, ask questions, and get insights instantly.
    """)

# --- Main Content ---
# Title
st.markdown('<div class="main-header">Opus – Financial Document Analysis</div>', unsafe_allow_html=True)

# Upload Section
st.subheader("Upload Documents")

st.markdown("""
<div class="info-box">
Upload your financial documents to analyze. Opus supports PDF and TXT files.
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
            
            # Show success message and document details
            st.markdown(f"""
            <div class="success-box">
            Successfully processed {len(uploaded_files)} document(s)!
            {" Documents processed successfully." if save_to_library else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Show document details
            for doc in file_details:
                st.markdown(f"**{doc['name']}** ({doc['size']})")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

    # Document clustering if enabled
    if show_clustering:
        st.markdown("### Document Structure Analysis")
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
                            st.markdown("### Document Topics")
                            
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

# In the document library section of the Streamlit app
if show_document_library:
    st.markdown("---")
    st.header("Document Library")
    
    # Add document management options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Refresh Document List"):
            st.session_state.document_list = []  # Clear the cached list
            
    with col2:
        if st.button("Clear All Documents"):
            confirm = st.checkbox("Are you sure? This cannot be undone.", value=False)
            if confirm:
                try:
                    # Delete the collection and recreate it
                    st.session_state.pipeline.store.client.delete_collection(
                        collection_name=st.session_state.pipeline.store.collection_name
                    )
                    # Recreate the collection
                    embedding_dim = st.session_state.pipeline.embedder.embedding_dim
                    st.session_state.pipeline.store.create_collection(embedding_dim)
                    st.success("All documents have been removed from the library.")
                    st.session_state.document_list = []  # Clear the cached list
                except Exception as e:
                    st.error(f"Error clearing documents: {str(e)}")
    
    # List documents in the library
    try:
        # Cache document list in session state to avoid repeated queries
        if "document_list" not in st.session_state:
            # Query for unique document IDs
            results = st.session_state.pipeline.store.client.scroll(
                collection_name=st.session_state.pipeline.store.collection_name,
                limit=1000,
                with_payload=True
            )
            
            # Extract unique documents
            documents = {}
            if results and results.points:
                for point in results.points:
                    if 'document_id' in point.payload and 'document_name' in point.payload:
                        doc_id = point.payload['document_id']
                        doc_name = point.payload['document_name']
                        if doc_id not in documents:
                            documents[doc_id] = {
                                'name': doc_name,
                                'id': doc_id,
                                'chunk_count': 1
                            }
                        else:
                            documents[doc_id]['chunk_count'] += 1
            
            st.session_state.document_list = list(documents.values())
        
        # Display documents
        if not st.session_state.document_list:
            st.info("No documents in the library yet. Upload documents to add them.")
        else:
            st.subheader(f"Found {len(st.session_state.document_list)} Documents")
            
            # Create a grid layout for documents
            cols = st.columns(3)
            for i, doc in enumerate(st.session_state.document_list):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div class="doc-card">
                            <h4>{doc['name']}</h4>
                            <p>Chunks: {doc['chunk_count']}</p>
                            <p>ID: {doc['id']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a delete button for each document
                        if st.button(f"Delete", key=f"del_{doc['id']}"):
                            try:
                                deleted = st.session_state.pipeline.store.delete_document(doc['id'])
                                if deleted:
                                    st.success(f"Deleted {doc['name']}")
                                    # Remove from the list 
                                    st.session_state.document_list = [d for d in st.session_state.document_list if d['id'] != doc['id']]
                                    st.experimental_rerun()
                                else:
                                    st.error(f"Failed to delete {doc['name']}")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        
    except Exception as e:
        st.error(f"Error loading document library: {str(e)}")

# QA Section - Primary Feature
st.markdown("---")
st.subheader("Ask a Question About Your Documents")

if not uploaded_files:
    st.markdown("""
    <div class="warning-box">
    Please upload at least one document to start asking questions.
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
                st.markdown("### Answer")
                
                # Show search information if available
                if "search_performed" in result and result["search_performed"]:
                    if result.get("search_successful", False):
                        st.success(f"Enhanced with current market information")
                        
                        # Display document date estimate if available
                        if "document_date_estimate" in result and result["document_date_estimate"] and result["document_date_estimate"].lower() != "unknown":
                            st.info(f"Document information from approximately {result['document_date_estimate']}")
                            
                        st.markdown(f"*Search query: \"{result.get('search_query', query)}\"*")
                    else:
                        st.warning(f"Attempted to find current information but encountered an issue: {result.get('search_error', 'Unknown error')}")
                
                # Display the answer in a styled box
                st.markdown(f"""
                <div class="answer-box">
                {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if result.get("sources"):
                    st.markdown("### Sources")
                    
                    # Group sources by type
                    document_sources = [src for src in result["sources"] if src.get("source_type", "") != "web_search"]
                    web_sources = [src for src in result["sources"] if src.get("source_type", "") == "web_search"]
                    
                    # Display unified source container
                    with st.container():
                        # Display all document sources in a single dropdown to hide similarities
                        if document_sources:
                            with st.expander("Document Sources (Click to view)"):
                                for i, src in enumerate(document_sources, 1):
                                    st.markdown(f"**Source {i} – {src['document_name']}**")
                                    st.markdown("**Content:**")
                                    
                                    # Use styled content display
                                    st.markdown(f"""
                                    <div class="source-content">
                                    {src["text"].replace('<', '&lt;').replace('>', '&gt;')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add a separator between sources
                                    if i < len(document_sources):
                                        st.markdown("---")
                        
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
                                export_text += f"\nSource {i}: {src['document_name']}\n"
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
                                file_name="opus_results.txt",
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
                                file_name="opus_results.json",
                                mime="application/json"
                            )
                
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 20px 0;">
Opus - Financial Document Analysis System | RAG-powered insights
</div>
""", unsafe_allow_html=True)