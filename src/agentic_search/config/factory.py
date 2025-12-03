"""Factory functions for creating pipeline components."""

import logging
from typing import Dict, Optional, Any

from agentic_search.config.settings import Settings
from agentic_search.core.interfaces import (
    BaseLLM,
    BaseEmbedding,
    BaseRetriever,
    BaseRouter,
    BaseCache,
)
from agentic_search.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def create_components(
    settings: Optional[Settings] = None,
    enable_cache: bool = True,
    enable_web_search: bool = True,
) -> Dict[str, Any]:
    """Create all pipeline components from settings.

    Args:
        settings: Configuration settings (uses defaults if None).
        enable_cache: Whether to create cache component.
        enable_web_search: Whether to create web search retriever.

    Returns:
        Dictionary with all created components.
    """
    settings = settings or Settings()
    settings.validate()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Creating pipeline components...")

    # Create embedding provider (shared across components)
    # Force CPU to avoid MPS segfault on Apple Silicon
    from agentic_search.embeddings import SentenceTransformerEmbedding
    embedding = SentenceTransformerEmbedding(model_name=settings.embedding_model, device="cpu")
    logger.info(f"Created embedding provider: {embedding.name}")

    # Create LLM provider
    from agentic_search.llm import OpenAILLM
    llm = OpenAILLM(
        api_key=settings.openai_api_key,
        model=settings.openai_rag_model,
    )
    logger.info(f"Created LLM provider: {llm.name}/{llm.model}")

    # Create retrievers
    from agentic_search.retrievers import QdrantRetriever, WebSearchRetriever
    retrievers: Dict[str, BaseRetriever] = {}

    # Qdrant 10-K retriever
    try:
        tenk_retriever = QdrantRetriever(
            embedding=embedding,
            collection_name=settings.tenk_collection,
            qdrant_path=settings.qdrant_path,
            source_type="10k_data",
        )
        retrievers["qdrant_10k_data"] = tenk_retriever
        logger.info(f"Created retriever: qdrant_10k_data")
    except Exception as e:
        logger.warning(f"Failed to create 10K retriever: {e}")

    # Qdrant OpenAI docs retriever
    try:
        openai_retriever = QdrantRetriever(
            embedding=embedding,
            collection_name=settings.openai_collection,
            qdrant_path=settings.qdrant_path,
            source_type="openai_docs",
        )
        retrievers["qdrant_opnai_data"] = openai_retriever
        logger.info(f"Created retriever: qdrant_opnai_data")
    except Exception as e:
        logger.warning(f"Failed to create OpenAI docs retriever: {e}")

    # Web search retriever
    if enable_web_search:
        web_retriever = WebSearchRetriever(
            max_results=settings.max_search_results,
            timeout=settings.web_search_timeout,
        )
        retrievers["web_search"] = web_retriever
        logger.info(f"Created retriever: web_search")

    # Create router
    from agentic_search.routers import RuleBasedRouter, LLMRouter, CompositeRouter

    rule_router = RuleBasedRouter(local_companies=settings.local_companies)
    llm_router = LLMRouter(llm=OpenAILLM(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    ))
    router = CompositeRouter(rule_router=rule_router, llm_router=llm_router)
    logger.info(f"Created router: {router.name}")

    # Create cache
    cache: Optional[BaseCache] = None
    if enable_cache:
        if settings.cache_backend == "redis":
            from agentic_search.cache import RedisSemanticCache
            cache = RedisSemanticCache(
                embedding=embedding,
                similarity_threshold=settings.cache_similarity_threshold,
                max_size=settings.cache_max_size,
                redis_url=settings.redis_url,
                redis_db=settings.redis_db,
                ttl_seconds=settings.redis_ttl_seconds,
            )
            logger.info(f"Created cache: {cache.name} (Redis backend)")
        else:
            from agentic_search.cache import SemanticCache
            cache = SemanticCache(
                embedding=embedding,
                similarity_threshold=settings.cache_similarity_threshold,
                max_size=settings.cache_max_size,
                cache_file=settings.cache_file,
            )
            logger.info(f"Created cache: {cache.name} (JSON backend)")

    return {
        "settings": settings,
        "embedding": embedding,
        "llm": llm,
        "retrievers": retrievers,
        "router": router,
        "cache": cache,
    }


def create_pipeline(
    settings: Optional[Settings] = None,
    enable_cache: bool = True,
    enable_web_search: bool = True,
) -> RAGPipeline:
    """Create a fully configured RAG pipeline.

    This is the recommended way to create a pipeline for production use.

    Args:
        settings: Configuration settings (uses defaults if None).
        enable_cache: Whether to enable semantic caching.
        enable_web_search: Whether to enable web search.

    Returns:
        Configured RAGPipeline instance.

    Example:
        # Using environment variables
        pipeline = create_pipeline()

        # With custom settings
        settings = Settings(openai_api_key="sk-...")
        pipeline = create_pipeline(settings=settings)

        # Without cache
        pipeline = create_pipeline(enable_cache=False)

        # Search
        result = pipeline.search("What is Lyft's revenue?")
        print(result.response)
    """
    components = create_components(
        settings=settings,
        enable_cache=enable_cache,
        enable_web_search=enable_web_search,
    )

    pipeline = RAGPipeline(
        llm=components["llm"],
        router=components["router"],
        retrievers=components["retrievers"],
        cache=components["cache"],
    )

    logger.info("RAG Pipeline created successfully")
    return pipeline
