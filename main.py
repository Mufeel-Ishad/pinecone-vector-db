"""
Pinecone Vector Database Implementation
This script sets up a Pinecone index, inserts article embeddings, and queries for closest matches.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Articles to be embedded
ARTICLES = [
    "AI is revolutionizing industries by enabling automation and improving efficiency.",
    "Quantum computing promises to solve complex problems that classical computers cannot.",
    "The future of blockchain technology includes decentralized finance and enhanced security."
]

INDEX_NAME = "article-index"
DIMENSION = 1536  # OpenAI text-embedding-ada-002 dimension
METRIC = "cosine"


def initialize_pinecone():
    """Initialize Pinecone with API key."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=api_key)
    return pc


def create_index(pc: Pinecone):
    """Create a new Pinecone index if it doesn't exist."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists. Using existing index.")
        return pc.Index(INDEX_NAME)
    
    print(f"Creating new index '{INDEX_NAME}'...")
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
    print("Waiting for index to be ready...")
    import time
    while INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        time.sleep(1)
    
    return pc.Index(INDEX_NAME)


def generate_embeddings(texts):
    """Generate embeddings using OpenAI's text-embedding-ada-002 model."""
    print("Generating embeddings...")
    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def insert_embeddings(index, articles, embeddings):
    """Insert embeddings into Pinecone index."""
    print("Inserting embeddings into Pinecone...")
    vectors = []
    for i, (article, embedding) in enumerate(zip(articles, embeddings)):
        vectors.append({
            "id": f"article-{i}",
            "values": embedding,
            "metadata": {"text": article}
        })
    
    # Insert in batches
    index.upsert(vectors=vectors)
    print(f"Successfully inserted {len(vectors)} vectors.")


def query_index(index, query_text, top_k=3):
    """Query the index for closest matches."""
    print(f"\nQuerying: '{query_text}'")
    
    # Generate embedding for the query
    query_embedding = generate_embeddings([query_text])[0]
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return results


def display_results(results):
    """Display query results."""
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    
    for i, match in enumerate(results.matches, 1):
        print(f"\n{i}. Article ID: {match.id}")
        print(f"   Similarity Score: {match.score:.4f}")
        print(f"   Text: {match.metadata.get('text', 'N/A')}")
    
    print("\n" + "="*60)


def delete_index(pc: Pinecone):
    """Delete the Pinecone index."""
    print(f"\nDeleting index '{INDEX_NAME}'...")
    try:
        pc.delete_index(INDEX_NAME)
        print(f"Index '{INDEX_NAME}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting index: {e}")


def main():
    """Main function to run the vector database operations."""
    try:
        # Initialize Pinecone
        pc = initialize_pinecone()
        
        # Create index
        index = create_index(pc)
        
        # Generate embeddings
        embeddings = generate_embeddings(ARTICLES)
        
        # Insert embeddings
        insert_embeddings(index, ARTICLES, embeddings)
        
        # Wait a moment for indexing to complete
        import time
        print("\nWaiting for vectors to be indexed...")
        time.sleep(2)
        
        # Query the index
        query = "What is the future of AI?"
        results = query_index(index, query, top_k=3)
        
        # Display results
        display_results(results)
        
        # Cleanup - delete the index
        print("\n" + "="*60)
        response = input("\nDo you want to delete the index? (yes/no): ").strip().lower()
        if response == "yes":
            delete_index(pc)
        else:
            print(f"Index '{INDEX_NAME}' preserved. You can delete it later.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

