"""RAG Pipeline - Orchestrates the complete search and generation flow."""

import logging
import re
import time
from typing import List, Dict, Any, Optional

from agentic_search.core.interfaces import (
    BaseLLM,
    BaseRouter,
    BaseRetriever,
    BaseCache,
)
from agentic_search.core.models import SearchResult, Document, QueryIntent
from agentic_search.core.exceptions import AgenticSearchError

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline with dependency injection.

    Orchestrates the complete search and generation flow:
    1. Check semantic cache for similar queries
    2. Route query to appropriate retrievers
    3. Retrieve relevant documents
    4. Generate response using LLM
    5. Store in cache for future use

    Example:
        # Using factory (recommended)
        pipeline = create_pipeline()

        # Or manual construction with dependency injection
        pipeline = RAGPipeline(
            llm=OpenAILLM(api_key="..."),
            router=CompositeRouter(...),
            retrievers={"10k": qdrant_retriever, "web": web_retriever},
            cache=SemanticCache(...),
        )

        result = pipeline.search("What is Lyft's revenue?")
        print(result.response)
    """

    RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context Documents:
{context}

User Question: {question}

Instructions:
- Provide a clear, comprehensive answer based ONLY on the information in the context above
- EXTRACT and PRESENT the actual facts, names, numbers, and details from each source
- For lists (like "top movies" or "best products"): compile the actual items mentioned across sources
- For news/current events: summarize what happened, who was involved, when, and key figures
- For financial data: be precise with numbers, percentages, and dates
- Synthesize information from multiple sources - combine and deduplicate
- You MAY cite sources at the end, but present the actual information directly
- NEVER say "check this link" or "visit this URL" - extract and present information directly
- If sources contain partial information, combine what you can find and note gaps

Answer:"""

    def __init__(
        self,
        llm: BaseLLM,
        router: BaseRouter,
        retrievers: Dict[str, BaseRetriever],
        cache: Optional[BaseCache] = None,
    ):
        """Initialize RAG pipeline with injected dependencies.

        Args:
            llm: LLM provider for generation.
            router: Query router for determining data sources.
            retrievers: Dictionary mapping retriever names to instances.
            cache: Optional semantic cache.
        """
        self._llm = llm
        self._router = router
        self._retrievers = retrievers
        self._cache = cache

        logger.info(
            f"RAG Pipeline initialized with: "
            f"LLM={llm.name}/{llm.model}, "
            f"Router={router.name}, "
            f"Retrievers={list(retrievers.keys())}, "
            f"Cache={'enabled' if cache else 'disabled'}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        allowed_retrievers: Optional[List[str]] = None,
    ) -> SearchResult:
        """Execute a complete search query.

        Args:
            query: User's question.
            top_k: Number of documents to retrieve.
            use_cache: Whether to use semantic cache.
            allowed_retrievers: Restrict to specific retrievers.

        Returns:
            SearchResult with response and metadata.
        """
        timing = {}
        start_time = time.time()

        # Validate input
        if not query or not query.strip():
            raise AgenticSearchError("Query cannot be empty")

        # Step 1: Check cache
        cache_start = time.time()
        cache_hit = False
        cache_metadata = {}

        if use_cache and self._cache:
            cached_response, cache_hit, cache_metadata = self._cache.lookup(query)
            timing["cache_lookup_ms"] = (time.time() - cache_start) * 1000

            if cache_hit:
                return SearchResult(
                    query=query,
                    response=cached_response,
                    sources=[],
                    routing_decision={"intent": "cached", "reason": "Cache hit"},
                    cache_hit=True,
                    cache_metadata=cache_metadata,
                    timing={"total_ms": (time.time() - start_time) * 1000, **timing},
                )

        # Step 2: Route query
        route_start = time.time()
        available = allowed_retrievers or list(self._retrievers.keys())
        routing_decision = self._router.route(query, available_retrievers=available)
        timing["routing_ms"] = (time.time() - route_start) * 1000

        logger.info(
            f"Routing: intent={routing_decision.intent.value}, "
            f"confidence={routing_decision.confidence:.0%}, "
            f"targets={routing_decision.target_retrievers}"
        )

        # Step 3: Retrieve documents
        retrieval_start = time.time()
        documents = self._retrieve_documents(query, routing_decision, top_k)
        timing["retrieval_ms"] = (time.time() - retrieval_start) * 1000

        # Step 4: Generate response
        generation_start = time.time()
        response = self._generate_response(query, documents)
        timing["generation_ms"] = (time.time() - generation_start) * 1000

        # Step 5: Store in cache
        if use_cache and self._cache and not cache_hit:
            cache_store_start = time.time()
            self._cache.store(
                question=query,
                response=response,
                metadata={
                    "routing": routing_decision.to_dict(),
                    "source_count": len(documents),
                },
            )
            timing["cache_store_ms"] = (time.time() - cache_store_start) * 1000

        timing["total_ms"] = (time.time() - start_time) * 1000

        # Convert documents to dict format for result
        sources = [doc.to_dict() for doc in documents]

        return SearchResult(
            query=query,
            response=response,
            sources=sources,
            routing_decision=routing_decision.to_dict(),
            cache_hit=cache_hit,
            cache_metadata=cache_metadata,
            timing=timing,
        )

    def _retrieve_documents(
        self,
        query: str,
        routing: Any,
        top_k: int,
    ) -> List[Document]:
        """Retrieve documents based on routing decision."""
        documents = []

        # Determine which retrievers to use
        target_retrievers = routing.target_retrievers
        if not target_retrievers:
            # Fallback based on intent
            intent_to_retrievers = {
                QueryIntent.LOCAL_10K: ["qdrant_10k_data"],
                QueryIntent.LOCAL_OPENAI: ["qdrant_opnai_data"],
                QueryIntent.WEB_SEARCH: ["web_search"],
                QueryIntent.HYBRID: ["qdrant_10k_data", "web_search"],
            }
            target_retrievers = intent_to_retrievers.get(routing.intent, [])

        # Calculate per-retriever limit for hybrid
        per_retriever_k = max(1, top_k // len(target_retrievers)) if target_retrievers else top_k

        # Retrieve from each target
        for retriever_name in target_retrievers:
            retriever = self._retrievers.get(retriever_name)
            if retriever:
                try:
                    docs = retriever.retrieve(query, top_k=per_retriever_k)
                    documents.extend(docs)
                    logger.debug(f"Retrieved {len(docs)} from {retriever_name}")
                except Exception as e:
                    logger.warning(f"Retriever {retriever_name} failed: {e}")

        logger.info(f"Total documents retrieved: {len(documents)}")
        return documents

    def _clean_content(self, text: str) -> str:
        """Clean content to remove formatting artifacts."""
        if not text:
            return text
        # Remove LaTeX-style formatting
        text = re.sub(r'\$([^$]+)\$', r'\1', text)
        # Remove invisible characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate response using LLM with retrieved context."""
        if not documents:
            return (
                "I couldn't find relevant information to answer your question. "
                "Please try rephrasing or ask about specific topics in the available data."
            )

        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = self._clean_content(doc.content[:2000])
            source_type = doc.source

            if source_type == "web_search":
                url = doc.metadata.get("url", "")
                title = doc.metadata.get("title", "")
                context_parts.append(
                    f"[Source {i}] WEB SEARCH RESULT\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content}"
                )
            else:
                context_parts.append(
                    f"[Source {i}] {source_type.upper()} (relevance: {doc.score:.2f})\n"
                    f"{content}"
                )

        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        try:
            response = self._llm.generate(prompt, temperature=0.3, max_tokens=1000)
            return response.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {e}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats().to_dict()
        return {"enabled": False}

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self._cache:
            self._cache.clear()

    def get_retriever_info(self) -> List[Dict[str, Any]]:
        """Get information about available retrievers."""
        return [r.get_info() for r in self._retrievers.values()]
