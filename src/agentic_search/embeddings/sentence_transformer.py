"""SentenceTransformer embedding provider implementation."""

import logging
from typing import List, Optional
import numpy as np

from agentic_search.core.interfaces import BaseEmbedding
from agentic_search.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """SentenceTransformer-based embedding provider.

    Supports any model from the sentence-transformers library, including:
    - nomic-ai/nomic-embed-text-v1.5 (768 dim)
    - all-MiniLM-L6-v2 (384 dim)
    - all-mpnet-base-v2 (768 dim)

    Example:
        embedding = SentenceTransformerEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        vectors = embedding.encode(["Hello world", "How are you?"])
        print(vectors.shape)  # (2, 768)
    """

    # Class-level model cache for singleton behavior
    _model_cache: dict = {}

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code: bool = True,
        device: Optional[str] = None,
        cache_model: bool = True,
    ):
        """Initialize SentenceTransformer embedding provider.

        Args:
            model_name: HuggingFace model identifier.
            trust_remote_code: Whether to trust remote code (needed for some models).
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            cache_model: Whether to cache the model for reuse.
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._cache_model = cache_model
        self._model = None
        self._dimension: Optional[int] = None
        self._device = device

        # Load model lazily on first use, or reuse cached model
        if cache_model and model_name in self._model_cache:
            self._model = self._model_cache[model_name]
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Reusing cached embedding model: {model_name}")

    def _load_model(self):
        """Lazily load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._model_name}")

            # Force CPU to avoid MPS segfault on Apple Silicon
            device = self._device or "cpu"
            kwargs = {"trust_remote_code": self._trust_remote_code, "device": device}

            self._model = SentenceTransformer(self._model_name, **kwargs)
            self._dimension = self._model.get_sentence_embedding_dimension()

            if self._cache_model:
                self._model_cache[self._model_name] = self._model

            logger.info(
                f"Loaded embedding model: {self._model_name} "
                f"(dimension: {self._dimension})"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Failed to load model {self._model_name}: {e}") from e

    @property
    def name(self) -> str:
        return "sentence_transformer"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode.
            normalize: Whether to normalize embeddings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            **kwargs: Additional encode parameters.

        Returns:
            NumPy array of shape (len(texts), dimension).

        Raises:
            EmbeddingError: If encoding fails.
        """
        self._load_model()

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                **kwargs,
            )

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise EmbeddingError(f"Encoding failed: {e}") from e

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._model_cache.clear()
        logger.info("Cleared embedding model cache")

    @classmethod
    def get_cached_models(cls) -> List[str]:
        """Get list of cached model names."""
        return list(cls._model_cache.keys())
