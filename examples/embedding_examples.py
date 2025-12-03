#!/usr/bin/env python
"""
Examples of using the Embedding API for different data sources.

Prerequisites:
    1. Start the API server:
       uvicorn api:app --reload --port 8000

    2. (Optional) For MongoDB examples:
       pip install pymongo

Usage:
    python examples/embedding_examples.py
"""

import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000"


# ============================================================
# 1. SIMPLE TEXT EMBEDDING
# ============================================================

def embed_single_text():
    """Generate embedding for a single text."""
    print("\n=== 1. Single Text Embedding ===")

    response = requests.post(
        f"{API_BASE}/embed",
        json={"text": "What is machine learning?"}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Dimension: {data['dimension']}")
        print(f"Embedding (first 5 values): {data['embedding'][:5]}")
    else:
        print(f"Error: {response.text}")


def embed_batch_texts():
    """Generate embeddings for multiple texts at once."""
    print("\n=== 2. Batch Text Embedding ===")

    texts = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning",
    ]

    response = requests.post(
        f"{API_BASE}/embed/batch",
        json={"texts": texts}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Generated {data['count']} embeddings")
        print(f"Each embedding has {data['dimension']} dimensions")
    else:
        print(f"Error: {response.text}")


# ============================================================
# 2. PDF EMBEDDING
# ============================================================

def embed_pdf(pdf_path: str):
    """
    Extract text from PDF, chunk it, and generate embeddings.

    Args:
        pdf_path: Path to the PDF file
    """
    print("\n=== 3. PDF Embedding ===")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"PDF not found: {pdf_path}")
        print("Creating a sample request instead...")

        # Show what the request would look like
        print("""
    curl -X POST http://localhost:8000/ingest/pdf \\
         -F "file=@your_document.pdf" \\
         -F "chunk_size=500" \\
         -F "chunk_overlap=50"
        """)
        return

    with open(pdf_file, "rb") as f:
        response = requests.post(
            f"{API_BASE}/ingest/pdf",
            files={"file": (pdf_file.name, f, "application/pdf")},
            data={"chunk_size": 500, "chunk_overlap": 50},
        )

    if response.status_code == 200:
        data = response.json()
        print(f"Created {data['chunks']} chunks")
        print(f"Embeddings shape: {data['embeddings_shape']}")
        print(f"Sample chunks:")
        for i, chunk in enumerate(data['sample_chunks']):
            print(f"  {i+1}. {chunk}")
    else:
        print(f"Error: {response.text}")


# ============================================================
# 3. TEXT CHUNKING AND EMBEDDING
# ============================================================

def embed_long_text():
    """Chunk a long text and generate embeddings for each chunk."""
    print("\n=== 4. Long Text Chunking ===")

    long_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing computer programs that can access data and use it
    to learn for themselves.

    The process begins with observations or data, such as examples, direct
    experience, or instruction. It looks for patterns in data so it can later
    make inferences based on the examples provided. The primary aim is to allow
    computers to learn automatically without human intervention.

    Deep learning is a subset of machine learning that uses neural networks with
    many layers. These deep neural networks attempt to simulate the behavior of
    the human brain in processing data for use in decision making.
    """

    response = requests.post(
        f"{API_BASE}/ingest/text",
        json={
            "text": long_text,
            "chunk_size": 300,
            "chunk_overlap": 50,
            "metadata": {"source": "example", "topic": "ML basics"},
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Created {data['chunks']} chunks")
        print(f"Embeddings shape: {data['embeddings_shape']}")
        print(f"Sample chunks:")
        for i, chunk in enumerate(data['sample_chunks']):
            print(f"  {i+1}. {chunk}")
    else:
        print(f"Error: {response.text}")


# ============================================================
# 4. STORE IN QDRANT
# ============================================================

def embed_and_store_in_qdrant():
    """Generate embeddings and store directly in Qdrant."""
    print("\n=== 5. Store in Qdrant ===")

    texts = [
        "Machine learning enables computers to learn from data.",
        "Neural networks are inspired by the human brain.",
        "Deep learning uses multiple layers of neural networks.",
    ]

    metadata = [
        {"source": "textbook", "chapter": 1},
        {"source": "textbook", "chapter": 2},
        {"source": "textbook", "chapter": 3},
    ]

    response = requests.post(
        f"{API_BASE}/ingest/qdrant",
        json={
            "texts": texts,
            "collection_name": "ml_examples",
            "metadata": metadata,
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Collection: {data['collection']}")
        print(f"Points added: {data['points_added']}")
        print(f"Total points: {data['total_points']}")
    else:
        print(f"Error: {response.text}")


# ============================================================
# 5. MONGODB DATA EMBEDDING
# ============================================================

def embed_mongodb_data():
    """
    Example: Fetch data from MongoDB and create embeddings.

    This shows the pattern for embedding MongoDB documents.
    """
    print("\n=== 6. MongoDB Data Embedding ===")

    # Check if pymongo is installed
    try:
        from pymongo import MongoClient
    except ImportError:
        print("pymongo not installed. Showing example code instead...")
        print("""
    # Install: pip install pymongo

    from pymongo import MongoClient

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["your_database"]
    collection = db["your_collection"]

    # Fetch documents
    documents = collection.find({"status": "active"}).limit(100)

    # Prepare texts for embedding
    texts = []
    metadata = []
    for doc in documents:
        # Combine fields you want to embed
        text = f"{doc.get('title', '')} {doc.get('description', '')}"
        texts.append(text)
        metadata.append({
            "mongo_id": str(doc["_id"]),
            "title": doc.get("title", ""),
        })

    # Call the embedding API
    response = requests.post(
        "http://localhost:8000/ingest/qdrant",
        json={
            "texts": texts,
            "collection_name": "mongodb_docs",
            "metadata": metadata,
        }
    )
    print(response.json())
        """)
        return

    # Example with actual MongoDB connection
    print("Connecting to MongoDB...")

    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.server_info()  # Test connection

        # Example: fetch and embed documents
        db = client["example_db"]
        collection = db["documents"]

        # Fetch documents
        documents = list(collection.find().limit(10))

        if not documents:
            print("No documents found. Insert some first.")
            return

        # Prepare for embedding
        texts = []
        metadata = []

        for doc in documents:
            # Combine text fields
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            texts.append(text)
            metadata.append({
                "mongo_id": str(doc["_id"]),
                "title": doc.get("title", ""),
            })

        # Call embedding API
        response = requests.post(
            f"{API_BASE}/ingest/qdrant",
            json={
                "texts": texts,
                "collection_name": "mongodb_docs",
                "metadata": metadata,
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Embedded {data['points_added']} MongoDB documents")
            print(f"Stored in Qdrant collection: {data['collection']}")
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Make sure MongoDB is running on localhost:27017")


# ============================================================
# 6. PROGRAMMATIC USAGE (WITHOUT API)
# ============================================================

def embed_directly():
    """
    Example: Use the embedding model directly without the API.

    Useful for batch processing or when you don't need HTTP.
    """
    print("\n=== 7. Direct Python Usage (No API) ===")

    print("""
    # Add src to path
    import sys
    sys.path.insert(0, "src")

    from agentic_search.embeddings import SentenceTransformerEmbedding

    # Create embedding model
    model = SentenceTransformerEmbedding(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
    )

    # Single text
    embedding = model.encode_single("What is AI?")
    print(f"Shape: {embedding.shape}")  # (768,)

    # Batch texts
    texts = ["Hello", "World", "AI is amazing"]
    embeddings = model.encode(texts)
    print(f"Shape: {embeddings.shape}")  # (3, 768)

    # Store in Qdrant directly
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance

    client = QdrantClient(path="./qdrant_data")
    client.create_collection(
        collection_name="my_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": text})
        for i, (text, emb) in enumerate(zip(texts, embeddings))
    ]
    client.upsert(collection_name="my_docs", points=points)
    """)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Embedding API Examples")
    print("=" * 60)

    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code == 200:
            print(f"\nAPI is running: {response.json()}")
        else:
            print("\nAPI returned unexpected status")
    except requests.exceptions.ConnectionError:
        print(f"\nAPI not running at {API_BASE}")
        print("Start it with: uvicorn api:app --reload --port 8000")
        print("\nShowing example code instead...\n")
        embed_directly()
        return

    # Run examples
    embed_single_text()
    embed_batch_texts()
    embed_pdf("sample.pdf")  # Will show curl example if file doesn't exist
    embed_long_text()
    embed_and_store_in_qdrant()
    embed_mongodb_data()
    embed_directly()

    print("\n" + "=" * 60)
    print("Done! Check the API docs at http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
