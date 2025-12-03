#!/usr/bin/env python
"""
FastAPI server for embedding generation and document ingestion.

Run with: uvicorn api:app --reload --port 8000

Endpoints:
    POST /embed           - Generate embeddings for text
    POST /embed/batch     - Generate embeddings for multiple texts
    POST /ingest/pdf      - Extract text from PDF and create embeddings
    POST /ingest/text     - Chunk text and create embeddings
    POST /ingest/qdrant   - Store embeddings in Qdrant
    GET  /health          - Health check
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from typing import List, Optional
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import time

from agentic_search.embeddings import SentenceTransformerEmbedding
from agentic_search.config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Search API",
    description="API for RAG search, embeddings, and document ingestion",
    version="1.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons (lazy loaded)
_embedding_model = None
_settings = None
_cache = None
_pipeline = None


def get_embedding_model() -> SentenceTransformerEmbedding:
    """Get or create the embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            device="cpu",
        )
    return _embedding_model


def get_settings() -> Settings:
    """Get settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_pipeline():
    """Get or create the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from agentic_search.config.factory import create_pipeline
        logger.info("Creating RAG pipeline...")
        _pipeline = create_pipeline(
            settings=get_settings(),
            enable_cache=True,
            enable_web_search=True,
        )
        logger.info("RAG pipeline ready")
    return _pipeline


# ----- Request/Response Models -----

class EmbedRequest(BaseModel):
    """Request for single text embedding."""
    text: str = Field(..., description="Text to embed")
    normalize: bool = Field(True, description="Normalize the embedding vector")


class EmbedBatchRequest(BaseModel):
    """Request for batch text embedding."""
    texts: List[str] = Field(..., description="List of texts to embed")
    normalize: bool = Field(True, description="Normalize embedding vectors")


class EmbedResponse(BaseModel):
    """Response containing embedding."""
    embedding: List[float] = Field(..., description="768-dimensional embedding vector")
    dimension: int = Field(..., description="Embedding dimension")


class EmbedBatchResponse(BaseModel):
    """Response containing multiple embeddings."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    count: int = Field(..., description="Number of embeddings")
    dimension: int = Field(..., description="Embedding dimension")


class TextChunk(BaseModel):
    """A chunk of text with metadata."""
    content: str
    metadata: dict = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    """Request to chunk and embed text."""
    text: str = Field(..., description="Text to chunk and embed")
    chunk_size: int = Field(500, description="Characters per chunk")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    metadata: dict = Field(default_factory=dict, description="Metadata to attach")


class IngestResponse(BaseModel):
    """Response from ingestion."""
    chunks: int = Field(..., description="Number of chunks created")
    embeddings_shape: List[int] = Field(..., description="Shape of embeddings array")
    sample_chunks: List[str] = Field(..., description="First few chunk previews")


class QdrantIngestRequest(BaseModel):
    """Request to store embeddings in Qdrant."""
    texts: List[str] = Field(..., description="Texts to embed and store")
    collection_name: str = Field(..., description="Qdrant collection name")
    metadata: List[dict] = Field(default_factory=list, description="Metadata per text")


class QdrantIngestResponse(BaseModel):
    """Response from Qdrant ingestion."""
    collection: str
    points_added: int
    total_points: int


class SearchRequest(BaseModel):
    """Request for RAG search."""
    query: str = Field(..., description="Search query")
    use_cache: bool = Field(True, description="Whether to use semantic cache")


class SearchResponse(BaseModel):
    """Response from RAG search."""
    query: str
    response: str
    sources: List[dict] = Field(default_factory=list, description="List of source documents")
    cache_hit: bool
    cache_metadata: dict = Field(default_factory=dict)
    routing_decision: dict = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict, description="Timing breakdown in ms")


# ----- Endpoints -----

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": get_settings().embedding_model}


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Generate embedding for a single text.

    Example:
        curl -X POST http://localhost:8000/embed \
             -H "Content-Type: application/json" \
             -d '{"text": "What is machine learning?"}'
    """
    try:
        model = get_embedding_model()
        embedding = model.encode([request.text], normalize=request.normalize)

        return EmbedResponse(
            embedding=embedding[0].tolist(),
            dimension=len(embedding[0]),
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    """
    Generate embeddings for multiple texts.

    Example:
        curl -X POST http://localhost:8000/embed/batch \
             -H "Content-Type: application/json" \
             -d '{"texts": ["Hello world", "How are you?"]}'
    """
    try:
        model = get_embedding_model()
        embeddings = model.encode(request.texts, normalize=request.normalize)

        return EmbedBatchResponse(
            embeddings=[e.tolist() for e in embeddings],
            count=len(embeddings),
            dimension=len(embeddings[0]) if embeddings.size > 0 else 0,
        )
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    """
    Chunk text and generate embeddings.

    Use this to prepare text for vector storage.
    Returns chunks and their embeddings.
    """
    try:
        # Simple chunking
        chunks = chunk_text(
            request.text,
            chunk_size=request.chunk_size,
            overlap=request.chunk_overlap,
        )

        # Generate embeddings
        model = get_embedding_model()
        embeddings = model.encode(chunks, normalize=True)

        return IngestResponse(
            chunks=len(chunks),
            embeddings_shape=list(embeddings.shape),
            sample_chunks=[c[:100] + "..." if len(c) > 100 else c for c in chunks[:3]],
        )
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    Extract text from PDF, chunk it, and generate embeddings.

    Example:
        curl -X POST http://localhost:8000/ingest/pdf \
             -F "file=@document.pdf" \
             -F "chunk_size=500"
    """
    try:
        import fitz  # PyMuPDF

        # Read PDF
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")

        # Extract text from all pages
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Chunk and embed
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        model = get_embedding_model()
        embeddings = model.encode(chunks, normalize=True)

        return IngestResponse(
            chunks=len(chunks),
            embeddings_shape=list(embeddings.shape),
            sample_chunks=[c[:100] + "..." if len(c) > 100 else c for c in chunks[:3]],
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PyMuPDF not installed. Run: pip install pymupdf"
        )
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/qdrant", response_model=QdrantIngestResponse)
async def ingest_to_qdrant(request: QdrantIngestRequest):
    """
    Generate embeddings and store directly in Qdrant.

    Example:
        curl -X POST http://localhost:8000/ingest/qdrant \
             -H "Content-Type: application/json" \
             -d '{
                 "texts": ["Document 1 text", "Document 2 text"],
                 "collection_name": "my_collection",
                 "metadata": [{"source": "doc1"}, {"source": "doc2"}]
             }'
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, VectorParams, Distance

        settings = get_settings()
        model = get_embedding_model()

        # Generate embeddings
        embeddings = model.encode(request.texts, normalize=True)

        # Connect to Qdrant
        client = QdrantClient(path=settings.qdrant_path)

        # Create collection if not exists
        collections = [c.name for c in client.get_collections().collections]
        if request.collection_name not in collections:
            client.create_collection(
                collection_name=request.collection_name,
                vectors_config=VectorParams(
                    size=model.dimension,
                    distance=Distance.COSINE,
                ),
            )

        # Get current count for IDs
        collection_info = client.get_collection(request.collection_name)
        start_id = collection_info.points_count

        # Prepare points
        points = []
        for i, (text, embedding) in enumerate(zip(request.texts, embeddings)):
            metadata = request.metadata[i] if i < len(request.metadata) else {}
            points.append(
                PointStruct(
                    id=start_id + i,
                    vector=embedding.tolist(),
                    payload={"content": text, **metadata},
                )
            )

        # Upsert to Qdrant
        client.upsert(collection_name=request.collection_name, points=points)

        # Get updated count
        collection_info = client.get_collection(request.collection_name)

        return QdrantIngestResponse(
            collection=request.collection_name,
            points_added=len(points),
            total_points=collection_info.points_count,
        )
    except Exception as e:
        logger.error(f"Qdrant ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform RAG search using the full pipeline.

    This endpoint:
    1. Checks semantic cache for similar queries
    2. Routes query to appropriate retriever (Qdrant/web)
    3. Generates response using LLM with retrieved context
    4. Caches the response for future queries

    Example:
        curl -X POST http://localhost:8000/search \
             -H "Content-Type: application/json" \
             -d '{"query": "What is Lyft revenue?"}'
    """
    try:
        pipeline = get_pipeline()

        # Search with cache option
        result = pipeline.search(request.query, use_cache=request.use_cache)

        return SearchResponse(
            query=request.query,
            response=result.response,
            sources=result.sources,
            cache_hit=result.cache_hit,
            cache_metadata=result.cache_metadata,
            routing_decision=result.routing_decision,
            timing=result.timing,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def cache_stats():
    """Get semantic cache statistics."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear the semantic cache."""
    try:
        pipeline = get_pipeline()
        pipeline.clear_cache()
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----- Helper Functions -----

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Clean text
    text = text.strip()
    if not text:
        return []

    # Split into sentences (simple approach)
    sentences = text.replace('\n', ' ').split('. ')

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Add period back if it was removed
        if not sentence.endswith('.'):
            sentence += '.'

        # If adding this sentence exceeds chunk size, save current and start new
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ----- Startup Event -----

@app.on_event("startup")
async def startup_event():
    """Pre-load the embedding model on startup."""
    logger.info("Starting Embedding API server...")
    # Lazy load - model will be loaded on first request
    # Uncomment to pre-load:
    # get_embedding_model()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
