# Pinecone Vector Database Implementation

A simple vector database implementation using Pinecone to store and query article embeddings generated with OpenAI's text-embedding-ada-002 model.

## Features

- ✅ Pinecone index creation with cosine similarity
- ✅ OpenAI embedding generation for articles
- ✅ Vector insertion into Pinecone
- ✅ Semantic search queries
- ✅ Interactive Streamlit UI (bonus feature)
- ✅ Index cleanup functionality

## Prerequisites

- Python 3.8 or higher
- Pinecone API key ([Get one here](https://www.pinecone.io/))
- OpenAI API key ([Get one here](https://platform.openai.com/))

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   
   **Option A - Using setup script (recommended):**
   ```bash
   python setup_env.py
   ```
   
   **Option B - Manual setup:**
   - Create a `.env` file in the project directory
   - Add your API keys:
     ```
     PINECONE_API_KEY=your_pinecone_api_key_here
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

### Option 1: Command Line Script

Run the main script to:
- Create the Pinecone index
- Insert article embeddings
- Query with "What is the future of AI?"
- Optionally delete the index

```bash
python main.py
```

### Option 2: Interactive UI (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

The UI allows you to:
- Initialize the index and insert articles
- Enter custom queries
- View search results with similarity scores
- Delete the index when done

## Articles Indexed

The following articles are embedded and stored in the vector database:

1. "AI is revolutionizing industries by enabling automation and improving efficiency."
2. "Quantum computing promises to solve complex problems that classical computers cannot."
3. "The future of blockchain technology includes decentralized finance and enhanced security."

## Index Configuration

- **Name:** `article-index`
- **Dimension:** 1536 (OpenAI text-embedding-ada-002)
- **Metric:** Cosine similarity
- **Cloud:** AWS (Serverless)
- **Region:** us-east-1

## Query Example

Default query: "What is the future of AI?"

This query will retrieve the top 3 most similar articles based on semantic similarity.

## Project Structure

```
pinecone-vector-db/
├── main.py              # Main command-line script
├── app.py               # Streamlit interactive UI
├── setup_env.py         # Environment setup helper
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .env                # Environment variables (create this)
```

## How It Works

1. **Initialization:** The script connects to Pinecone using your API key
2. **Index Creation:** Creates a new index named "article-index" with 1536 dimensions
3. **Embedding Generation:** Uses OpenAI's text-embedding-ada-002 to convert articles into vectors
4. **Vector Storage:** Inserts embeddings into Pinecone with metadata
5. **Query Processing:** Converts query text to embedding and searches for similar vectors
6. **Results:** Returns top matches with similarity scores

## Cleanup

Both scripts provide options to delete the Pinecone index after use to free up resources. The index can also be deleted manually through the Pinecone dashboard.

## Notes

- The index is created in AWS us-east-1 region (serverless)
- Vectors are assigned IDs: `article-0`, `article-1`, `article-2`
- Each vector includes metadata with the original article text
- The index persists between runs until explicitly deleted
- First run will download the model (if using local alternatives)

## Troubleshooting

### Common Issues

1. **API Key Errors:**
   - Ensure your `.env` file is in the project root
   - Check that API keys are correctly formatted (no quotes needed)

2. **Index Already Exists:**
   - The script will use the existing index if found
   - Delete the index through Pinecone dashboard or use the cleanup function

3. **OpenAI Rate Limits:**
   - Check your OpenAI account quota
   - Wait for rate limit reset or upgrade your plan

4. **Dimension Mismatch:**
   - If you change embedding models, delete the old index first
   - Different models have different dimensions

## License

This project is provided as-is for educational and demonstration purposes.
