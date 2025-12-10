# RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI, Cohere embeddings, Qdrant vector database, and Google's Gemini 2.5 Flash model.

## Features

- Document indexing and retrieval
- Semantic search using Cohere embeddings
- Vector storage in Qdrant Cloud
- AI-powered responses using Gemini 2.5 Flash
- FastAPI backend with clean architecture

## Architecture

The system consists of several components:

- **Embedder**: Uses Cohere's `embed-english-v3.0` model to generate document and query embeddings
- **VectorDB**: Stores embeddings in Qdrant Cloud for efficient similarity search
- **LLM**: Google's Gemini 2.5 Flash for generating contextual responses
- **RAG Engine**: Orchestrates the entire pipeline - chunking, indexing, searching, and response generation

## Setup

1. Clone the repository
2. Install dependencies: `uv pip install fastapi uvicorn python-dotenv cohere qdrant-client google-generativeai pydantic`
3. Set up environment variables in `.env` file:

```bash
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
```

4. Place your documents in the `textbook/` folder

## API Endpoints

### POST /index
Index documents from a folder into the vector database.

Request:
```json
{
  "folder_path": "textbook"
}
```

Response:
```json
{
  "message": "Documents indexed successfully",
  "details": {
    "indexed_documents": 5,
    "total_chunks": 15,
    "sources": ["textbook/doc1.txt", "textbook/doc2.txt"]
  }
}
```

### POST /query
Query the RAG system to get an answer based on indexed documents.

Request:
```json
{
  "question": "What is artificial intelligence?",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence...",
  "sources": [
    {
      "text": "Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence...",
      "source": "textbook/ai_intro.txt",
      "title": "ai_intro"
    }
  ]
}
```

## Usage Example

1. Index documents:
```bash
curl -X POST "http://localhost:8000/index" \
     -H "Content-Type: application/json" \
     -d '{"folder_path": "textbook"}'
```

2. Query the system:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is machine learning?", "top_k": 3}'
```

## Local Development

Run the server locally:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Deployment

### Deploy on Render

The application can be deployed on Render using the following steps:

1. Fork this repository to your GitHub account
2. Go to https://render.com and sign up/in with your GitHub account
3. Create a new Web Service
4. Connect your forked repository
5. Set the runtime to Python
6. Use the following environment variables:
   - `COHERE_API_KEY`: Your Cohere API key
   - `QDRANT_URL`: Your Qdrant Cloud URL
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `GEMINI_API_KEY`: Your Google Gemini API key
7. The build and start commands are already configured in the runtime.txt file
8. Deploy the service

The application will be accessible at the provided Render URL.

### Local Development

To run locally:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.