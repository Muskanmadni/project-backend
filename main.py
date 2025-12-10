import os
import uuid
from typing import List, Dict
from pathlib import Path
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot System", 
              description="A Retrieval-Augmented Generation chatbot using Cohere embeddings, Qdrant vector store, and Gemini 2.5 Flash",
              version="1.0.0")


# Request/Response Models
class IndexRequest(BaseModel):
    folder_path: str = "textbook"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


class Embedder:
    """
    Embedder class for generating embeddings using Cohere API
    """
    
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        self.client = cohere.Client(api_key)
        self.model = "embed-english-v3.0"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        """
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings
        
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        """
        response = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query"
        )
        return response.embeddings[0]


class VectorDB:
    """
    VectorDB class for managing vector storage with Qdrant
    """
    
    def __init__(self):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = "documents"
        self._create_collection()
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to the collection"""
        # Generate IDs for the documents
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate embeddings
        embeddings = Embedder().embed_documents(texts)
        
        # Prepare points for insertion
        points = [
            models.PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "text": texts[i],
                    **metadatas[i]
                }
            )
            for i in range(len(texts))
        ]
        
        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents to the query"""
        embedder = Embedder()
        query_embedding = embedder.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]


class LLM:
    """
    LLM class for generating responses using Google's Gemini
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate_response(self, prompt: str) -> str:
        """Generate response based on the prompt"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response at the moment."


class RAGEngine:
    """
    Main RAG Engine that orchestrates the entire pipeline
    """
    
    def __init__(self):
        self.embedder = Embedder()
        self.vector_db = VectorDB()
        self.llm = LLM()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def index_folder(self, folder_path: str):
        """
        Index all text documents in a folder
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
        
        # Find all text files
        text_files = list(folder.glob("**/*.txt"))
        
        if not text_files:
            raise ValueError(f"No .txt files found in {folder_path}")
        
        documents_to_index = []
        metadatas = []
        
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Chunk the document
                chunks = self.chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents_to_index.append(chunk)
                    metadatas.append({
                        "source": str(file_path),
                        "chunk_id": i,
                        "title": file_path.stem
                    })
        
        # Add documents to vector DB
        ids = self.vector_db.add_documents(documents_to_index, metadatas)
        
        return {
            "indexed_documents": len(documents_to_index),
            "total_chunks": len(ids),
            "sources": list(set([meta["source"] for meta in metadatas]))
        }
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Process a query using the RAG pipeline
        """
        # Search for relevant documents
        search_results = self.vector_db.search(question, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": []
            }
        
        # Format context from retrieved documents
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result["text"])
            sources.append({
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "source": result["metadata"].get("source", "Unknown"),
                "title": result["metadata"].get("title", "Untitled")
            })
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt
        prompt = f"""Answer the question based on the provided context. If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response using LLM
        answer = self.llm.generate_response(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }


# Initialize RAG Engine
rag_engine = RAGEngine()


@app.post("/index", summary="Index documents from a folder")
async def index_documents(request: IndexRequest):
    """
    Index all text documents in the specified folder into the vector database.
    """
    try:
        result = rag_engine.index_folder(request.folder_path)
        return {
            "message": "Documents indexed successfully",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_rag(request: QueryRequest):
    """
    Query the RAG system to get an answer based on indexed documents.
    """
    try:
        result = rag_engine.query(request.question, top_k=request.top_k)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Health check endpoint")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy", "service": "RAG Chatbot System"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)