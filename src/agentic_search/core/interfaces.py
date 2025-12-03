"""Abstract base classes defining the plugin interfaces.

All extensible components implement these interfaces, enabling:
- Swappable LLM providers
- Pluggable document retrievers
- Custom query routers
- Alternative caching backends
- Different embedding models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from agentic_search.core.models import (
    Document,
    RoutingDecision,
    CacheStats,
    LLMResponse,
)


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Implement this interface to add support for new LLM providers
    like Anthropic, Cohere, local models, etc.

    Example:
        class AnthropicLLM(BaseLLM):
            def __init__(self, api_key: str, model: str = "claude-3-sonnet"):
                self.client = anthropic.Client(api_key=api_key)
                self.model = model

            def generate(self, prompt: str, **kwargs) -> LLMResponse:
                response = self.client.messages.create(...)
                return LLMResponse(content=response.content[0].text, ...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model identifier being used."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with generated content and metadata.
        """
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            prompt: The input prompt expecting JSON output.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Returns:
            Parsed JSON dictionary.
        """
        pass


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.

    Implement this interface to add support for different embedding models
    like OpenAI embeddings, Cohere, local sentence transformers, etc.

    Example:
        class OpenAIEmbedding(BaseEmbedding):
            def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
                self.client = openai.Client(api_key=api_key)
                self.model_name = model

            def encode(self, texts: List[str]) -> np.ndarray:
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                return np.array([e.embedding for e in response.data])
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the embedding provider name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode.
            **kwargs: Provider-specific parameters.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        pass

    def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """Encode a single text into an embedding.

        Args:
            text: Text to encode.
            **kwargs: Provider-specific parameters.

        Returns:
            NumPy array of shape (dimension,).
        """
        return self.encode([text], **kwargs)[0]


class BaseRetriever(ABC):
    """Abstract base class for document retrievers.

    Implement this interface to add support for different vector stores
    like Pinecone, Weaviate, ChromaDB, or custom data sources.

    Example:
        class PineconeRetriever(BaseRetriever):
            def __init__(self, index_name: str, embedding: BaseEmbedding):
                self.index = pinecone.Index(index_name)
                self.embedding = embedding

            def retrieve(self, query: str, top_k: int) -> List[Document]:
                vector = self.embedding.encode_single(query)
                results = self.index.query(vector=vector, top_k=top_k)
                return [Document(content=r.metadata["text"], ...) for r in results.matches]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the retriever name (used for routing)."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type (e.g., '10k_data', 'web_search')."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            **kwargs: Retriever-specific parameters.

        Returns:
            List of Document objects sorted by relevance.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get information about this retriever.

        Returns:
            Dictionary with retriever metadata.
        """
        return {
            "name": self.name,
            "source_type": self.source_type,
        }


class BaseRouter(ABC):
    """Abstract base class for query routers.

    Implement this interface to create custom routing strategies
    that determine which retrievers to use for a given query.

    Example:
        class KeywordRouter(BaseRouter):
            def __init__(self, keyword_map: Dict[str, List[str]]):
                self.keyword_map = keyword_map

            def route(self, query: str) -> RoutingDecision:
                for intent, keywords in self.keyword_map.items():
                    if any(kw in query.lower() for kw in keywords):
                        return RoutingDecision(intent=intent, ...)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the router name."""
        pass

    @abstractmethod
    def route(
        self,
        query: str,
        available_retrievers: Optional[List[str]] = None,
        **kwargs,
    ) -> RoutingDecision:
        """Route a query to appropriate retrievers.

        Args:
            query: The user's query.
            available_retrievers: List of available retriever names.
            **kwargs: Router-specific parameters.

        Returns:
            RoutingDecision indicating which retrievers to use.
        """
        pass


class BaseCache(ABC):
    """Abstract base class for caching backends.

    Implement this interface to add support for different caching strategies
    like Redis, PostgreSQL, or in-memory caches.

    Example:
        class RedisCache(BaseCache):
            def __init__(self, redis_url: str, embedding: BaseEmbedding):
                self.redis = redis.from_url(redis_url)
                self.embedding = embedding

            def lookup(self, question: str) -> Tuple[Optional[str], bool, Dict]:
                # Search Redis for similar questions
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the cache backend name."""
        pass

    @abstractmethod
    def lookup(
        self,
        question: str,
        **kwargs,
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Look up a question in the cache.

        Args:
            question: The question to look up.
            **kwargs: Cache-specific parameters.

        Returns:
            Tuple of (cached_response, is_hit, metadata).
            If no hit, cached_response is None and is_hit is False.
        """
        pass

    @abstractmethod
    def store(
        self,
        question: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Store a question-response pair in the cache.

        Args:
            question: The question.
            response: The response to cache.
            metadata: Optional metadata to store.
            **kwargs: Cache-specific parameters.
        """
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit rate, size, etc.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    def is_enabled(self) -> bool:
        """Check if caching is enabled.

        Returns:
            True if caching is enabled.
        """
        return True
