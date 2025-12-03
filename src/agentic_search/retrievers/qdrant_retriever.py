"""Qdrant vector database retriever implementation."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentic_search.core.interfaces import BaseRetriever, BaseEmbedding
from agentic_search.core.models import Document
from agentic_search.core.exceptions import RetrieverError

logger = logging.getLogger(__name__)


class QdrantRetriever(BaseRetriever):
    """Qdrant-based document retriever.

    Retrieves documents from a Qdrant vector database collection.

    Example:
        embedding = SentenceTransformerEmbedding()
        retriever = QdrantRetriever(
            embedding=embedding,
            collection_name="10k_data",
            qdrant_path="./qdrant_data",
        )
        docs = retriever.retrieve("What are the risk factors?", top_k=5)
    """

    # Class-level client cache for sharing across retrievers
    _client_cache: dict = {}

    def __init__(
        self,
        embedding: BaseEmbedding,
        collection_name: str,
        qdrant_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        source_type: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """Initialize Qdrant retriever.

        Args:
            embedding: Embedding provider for query encoding.
            collection_name: Name of the Qdrant collection.
            qdrant_path: Path to local Qdrant storage (for local mode).
            qdrant_url: URL for Qdrant server (for server mode).
            source_type: Override source type label (defaults to collection_name).
            client: Optional existing QdrantClient to reuse.
        """
        self._embedding = embedding
        self._collection_name = collection_name
        self._source_type = source_type or collection_name
        self._client = None

        # Initialize Qdrant client (reuse if provided or cached)
        try:
            from qdrant_client import QdrantClient

            if client is not None:
                self._client = client
                logger.info(f"Reusing existing Qdrant client for collection: {collection_name}")
            elif qdrant_url:
                cache_key = f"url:{qdrant_url}"
                if cache_key in self._client_cache:
                    self._client = self._client_cache[cache_key]
                    logger.info(f"Reusing cached Qdrant client for: {qdrant_url}")
                else:
                    self._client = QdrantClient(url=qdrant_url)
                    self._client_cache[cache_key] = self._client
                    logger.info(f"Connected to Qdrant server: {qdrant_url}")
            elif qdrant_path:
                path = Path(qdrant_path)
                if not path.exists():
                    raise RetrieverError(f"Qdrant path does not exist: {qdrant_path}")
                cache_key = f"path:{path}"
                if cache_key in self._client_cache:
                    self._client = self._client_cache[cache_key]
                    logger.info(f"Reusing cached Qdrant client for: {qdrant_path}")
                else:
                    self._client = QdrantClient(path=str(path))
                    self._client_cache[cache_key] = self._client
                    logger.info(f"Connected to local Qdrant: {qdrant_path}")
            else:
                raise RetrieverError("Either qdrant_path or qdrant_url must be provided")

            # Verify collection exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                logger.warning(
                    f"Collection '{collection_name}' not found. "
                    f"Available: {collection_names}"
                )

        except Exception as e:
            if "RetrieverError" in str(type(e)):
                raise
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise RetrieverError(f"Qdrant initialization failed: {e}") from e

    @property
    def name(self) -> str:
        return f"qdrant_{self._collection_name}"

    @property
    def source_type(self) -> str:
        return self._source_type

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Document]:
        """Retrieve relevant documents from Qdrant.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            score_threshold: Minimum similarity score (0-1).
            **kwargs: Additional search parameters.

        Returns:
            List of Document objects sorted by relevance.

        Raises:
            RetrieverError: If retrieval fails.
        """
        try:
            from qdrant_client.models import models

            # Encode query
            query_vector = self._embedding.encode_single(query)

            # Search Qdrant using query_points (new API in qdrant-client >= 1.10)
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Convert to Document objects
            documents = []
            for point in response.points:
                payload = point.payload or {}
                content = payload.get("content", payload.get("text", ""))
                metadata = {k: v for k, v in payload.items() if k not in ("content", "text")}

                documents.append(
                    Document(
                        content=content,
                        source=self._source_type,
                        score=point.score,
                        metadata=metadata,
                    )
                )

            logger.debug(
                f"Retrieved {len(documents)} documents from {self._collection_name}"
            )
            return documents

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            raise RetrieverError(f"Qdrant search failed: {e}") from e

    def get_info(self) -> Dict[str, Any]:
        """Get information about this retriever and collection."""
        info = {
            "name": self.name,
            "source_type": self.source_type,
            "collection_name": self._collection_name,
        }

        try:
            collection_info = self._client.get_collection(self._collection_name)
            info["points_count"] = collection_info.points_count
            info["vectors_count"] = collection_info.vectors_count
        except Exception as e:
            info["error"] = str(e)

        return info

    def get_collection_names(self) -> List[str]:
        """Get list of all collection names in Qdrant."""
        try:
            collections = self._client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
