"""Custom exceptions for the agentic search engine."""


class AgenticSearchError(Exception):
    """Base exception for all agentic search errors."""
    pass


class LLMError(AgenticSearchError):
    """Error in LLM provider operations."""
    pass


class RetrieverError(AgenticSearchError):
    """Error in document retrieval operations."""
    pass


class RouterError(AgenticSearchError):
    """Error in query routing operations."""
    pass


class CacheError(AgenticSearchError):
    """Error in cache operations."""
    pass


class EmbeddingError(AgenticSearchError):
    """Error in embedding operations."""
    pass


class ConfigurationError(AgenticSearchError):
    """Error in configuration or initialization."""
    pass
