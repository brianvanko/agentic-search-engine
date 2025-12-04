"""Configuration settings for the agentic search engine."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

# Load .env file
load_dotenv()


def _safe_int(value: str, default: int) -> int:
    """Safely convert string to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value: str, default: float) -> float:
    """Safely convert string to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value: str, default: bool) -> bool:
    """Safely convert string to bool."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _parse_cors_origins(value: str) -> List[str]:
    """Parse comma-separated CORS origins.

    Supports:
    - "*" for all origins
    - Comma-separated URLs: "http://localhost:3000,http://example.com"
    - Also includes 127.0.0.1 variants for localhost entries
    """
    if not value or value.strip() == "*":
        return ["*"]

    origins = []
    for origin in value.split(","):
        origin = origin.strip()
        if origin:
            origins.append(origin)
            # Add 127.0.0.1 variant for localhost entries
            if "localhost" in origin:
                origins.append(origin.replace("localhost", "127.0.0.1"))

    return origins if origins else ["*"]


@dataclass
class Settings:
    """Configuration settings with environment variable support.

    All settings can be overridden via environment variables or .env file.

    Example:
        # Using defaults
        settings = Settings()

        # With overrides
        settings = Settings(openai_api_key="sk-...", qdrant_path="./data")

        # From environment
        settings = Settings.from_env()
    """

    # OpenAI Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_rag_model: str = field(default_factory=lambda: os.getenv("OPENAI_RAG_MODEL", "gpt-4o-mini"))

    # Embedding Configuration
    # Use nomic-ai/nomic-embed-text-v1.5 - 768-dim embeddings, same as used for 10-K indexing
    # Requires PyTorch 2.1.x for Apple Silicon compatibility
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    )
    embedding_dimension: int = field(
        default_factory=lambda: _safe_int(os.getenv("EMBEDDING_DIMENSION", "768"), 768)
    )

    # Qdrant Configuration
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data"))
    qdrant_url: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_URL"))
    tenk_collection: str = field(default_factory=lambda: os.getenv("TENK_COLLECTION", "10k_data"))
    openai_collection: str = field(default_factory=lambda: os.getenv("OPENAI_COLLECTION", "opnai_data"))

    # Cache Configuration
    cache_backend: str = field(default_factory=lambda: os.getenv("CACHE_BACKEND", "json"))  # "json" or "redis"
    cache_file: str = field(default_factory=lambda: os.getenv("CACHE_FILE", "./semantic_cache.json"))
    cache_similarity_threshold: float = field(
        default_factory=lambda: _safe_float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.78"), 0.78)
    )
    cache_max_size: int = field(
        default_factory=lambda: _safe_int(os.getenv("CACHE_MAX_SIZE", "10000"), 10000)
    )
    enable_cache: bool = field(
        default_factory=lambda: _safe_bool(os.getenv("ENABLE_CACHE", "true"), True)
    )

    # Redis Cache Configuration
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_db: int = field(default_factory=lambda: _safe_int(os.getenv("REDIS_DB", "0"), 0))
    redis_ttl_seconds: Optional[int] = field(
        default_factory=lambda: _safe_int(os.getenv("REDIS_TTL_SECONDS", ""), 0) or None
    )

    # Web Search Configuration
    enable_web_search: bool = field(
        default_factory=lambda: _safe_bool(os.getenv("ENABLE_WEB_SEARCH", "true"), True)
    )
    web_search_timeout: int = field(
        default_factory=lambda: _safe_int(os.getenv("WEB_SEARCH_TIMEOUT", "10"), 10)
    )
    max_search_results: int = field(
        default_factory=lambda: _safe_int(os.getenv("MAX_SEARCH_RESULTS", "5"), 5)
    )

    # General Settings
    top_k_results: int = field(
        default_factory=lambda: _safe_int(os.getenv("TOP_K_RESULTS", "5"), 5)
    )

    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(
        default_factory=lambda: os.getenv("LOG_FORMAT", "standard")  # "standard" or "json"
    )
    log_json: bool = field(
        default_factory=lambda: _safe_bool(os.getenv("LOG_JSON", "false"), False)
    )

    # CORS Configuration
    # Comma-separated list of allowed origins, or "*" for all origins
    cors_origins: List[str] = field(
        default_factory=lambda: _parse_cors_origins(
            os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001")
        )
    )
    cors_allow_credentials: bool = field(
        default_factory=lambda: _safe_bool(os.getenv("CORS_ALLOW_CREDENTIALS", "true"), True)
    )

    # Router Configuration
    local_companies: List[str] = field(default_factory=lambda: ["lyft"])

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate settings and raise errors for missing required values."""
        from agentic_search.core.exceptions import ConfigurationError

        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")

        qdrant_path = Path(self.qdrant_path)
        if not self.qdrant_url and not qdrant_path.exists():
            raise ConfigurationError(f"Qdrant path does not exist: {self.qdrant_path}")

    def __post_init__(self):
        """Resolve relative paths to absolute."""
        # Make paths absolute relative to the module directory
        base_dir = Path(__file__).parent.parent.parent.parent

        if not Path(self.qdrant_path).is_absolute():
            self.qdrant_path = str(base_dir / self.qdrant_path)

        if not Path(self.cache_file).is_absolute():
            self.cache_file = str(base_dir / self.cache_file)


# Default settings instance
default_settings = Settings()
