"""
Streamlit UI for Pinecone Vector Database Querying
Interactive interface to query the article embeddings.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize session state
if "index_initialized" not in st.session_state:
    st.session_state.index_initialized = False
if "index" not in st.session_state:
    st.session_state.index = None
if "pc" not in st.session_state:
    st.session_state.pc = None

INDEX_NAME = "article-index"
DIMENSION = 1536
METRIC = "cosine"

ARTICLES = [
    "AI is revolutionizing industries by enabling automation and improving efficiency.",
    "Quantum computing promises to solve complex problems that classical computers cannot.",
    "The future of blockchain technology includes decentralized finance and enhanced security."
]


@st.cache_resource
def initialize_pinecone():
    """Initialize Pinecone with API key."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("PINECONE_API_KEY not found in environment variables")
        return None
    
    pc = Pinecone(api_key=api_key)
    return pc


@st.cache_resource
def get_openai_client():
    """Get OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment variables")
        return None
    return OpenAI(api_key=api_key)


def create_or_get_index(pc: Pinecone):
    """Create or get existing Pinecone index."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        return pc.Index(INDEX_NAME)
    
    # Create index
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    # Wait for index to be ready
    import time
    while INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        time.sleep(1)
    
    return pc.Index(INDEX_NAME)


def generate_embeddings(openai_client, texts):
    """Generate embeddings using OpenAI."""
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def initialize_index_with_data():
    """Initialize the index and insert article embeddings."""
    if st.session_state.index_initialized:
        return
    
    with st.spinner("Initializing Pinecone and inserting articles..."):
        # Initialize clients
        pc = initialize_pinecone()
        openai_client = get_openai_client()
        
        if not pc or not openai_client:
            return
        
        # Get or create index
        index = create_or_get_index(pc)
        
        # Check if index already has data
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            # Generate embeddings
            embeddings = generate_embeddings(openai_client, ARTICLES)
            
            # Insert embeddings
            vectors = []
            for i, (article, embedding) in enumerate(zip(ARTICLES, embeddings)):
                vectors.append({
                    "id": f"article-{i}",
                    "values": embedding,
                    "metadata": {"text": article}
                })
            
            index.upsert(vectors=vectors)
            
            # Wait for indexing
            import time
            time.sleep(2)
        
        st.session_state.pc = pc
        st.session_state.index = index
        st.session_state.index_initialized = True
        st.success("Index initialized and articles inserted!")


def query_index(query_text, top_k=3):
    """Query the index for closest matches."""
    if not st.session_state.index or not st.session_state.pc:
        st.error("Index not initialized. Please initialize first.")
        return None
    
    openai_client = get_openai_client()
    if not openai_client:
        return None
    
    # Generate query embedding
    query_embedding = generate_embeddings(openai_client, [query_text])[0]
    
    # Query Pinecone
    results = st.session_state.index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Pinecone Vector Database Query",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Pinecone Vector Database Query Interface")
    st.markdown("Query article embeddings using semantic search")
    
    # Initialize index
    if st.button("Initialize Index & Insert Articles", type="primary"):
        initialize_index_with_data()
    
    if not st.session_state.index_initialized:
        st.info("👆 Click the button above to initialize the index and insert articles.")
        st.markdown("### Articles to be indexed:")
        for i, article in enumerate(ARTICLES):
            st.markdown(f"{i+1}. {article}")
        return
    
    st.divider()
    
    # Query interface
    st.header("Query Interface")
    
    # Default query
    default_query = "What is the future of AI?"
    query = st.text_input(
        "Enter your query:",
        value=default_query,
        placeholder="Type your question here..."
    )
    
    top_k = st.slider("Number of results to retrieve:", min_value=1, max_value=10, value=3)
    
    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = query_index(query, top_k=top_k)
                
                if results:
                    st.divider()
                    st.header("Search Results")
                    
                    for i, match in enumerate(results.matches, 1):
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric("Score", f"{match.score:.4f}")
                            with col2:
                                st.markdown(f"**Article ID:** `{match.id}`")
                                st.markdown(f"**Text:** {match.metadata.get('text', 'N/A')}")
                            st.divider()
                else:
                    st.error("No results found or error occurred.")
        else:
            st.warning("Please enter a query.")
    
    st.divider()
    
    # Cleanup section
    st.header("Index Management")
    if st.button("🗑️ Delete Index", type="secondary"):
        if st.session_state.pc:
            try:
                st.session_state.pc.delete_index(INDEX_NAME)
                st.success(f"Index '{INDEX_NAME}' deleted successfully!")
                st.session_state.index_initialized = False
                st.session_state.index = None
                st.session_state.pc = None
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting index: {e}")
        else:
            st.error("Pinecone client not initialized.")


if __name__ == "__main__":
    main()

