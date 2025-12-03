"""Composite router that combines rule-based and LLM routing."""

import logging
from typing import List, Optional

from agentic_search.core.interfaces import BaseRouter
from agentic_search.core.models import RoutingDecision

logger = logging.getLogger(__name__)


class CompositeRouter(BaseRouter):
    """Composite router that tries rule-based routing first, then LLM.

    This is the recommended router for production use. It provides:
    - Fast routing for obvious cases (rule-based)
    - Flexible routing for complex cases (LLM)
    - Graceful fallback if LLM fails

    Example:
        rule_router = RuleBasedRouter(local_companies=["lyft"])
        llm_router = LLMRouter(llm=OpenAILLM(...))
        router = CompositeRouter(
            rule_router=rule_router,
            llm_router=llm_router,
        )
        decision = router.route("What is Lyft's revenue?")
    """

    def __init__(
        self,
        rule_router: BaseRouter,
        llm_router: Optional[BaseRouter] = None,
    ):
        """Initialize composite router.

        Args:
            rule_router: Fast rule-based router.
            llm_router: LLM router for fallback (optional).
        """
        self._rule_router = rule_router
        self._llm_router = llm_router

    @property
    def name(self) -> str:
        return "composite"

    def route(
        self,
        query: str,
        available_retrievers: Optional[List[str]] = None,
        **kwargs,
    ) -> RoutingDecision:
        """Route a query using rule-based first, then LLM fallback.

        Args:
            query: The user's query.
            available_retrievers: List of available retriever names.
            **kwargs: Additional parameters.

        Returns:
            RoutingDecision with routing information.
        """
        # Try rule-based routing first
        decision = self._rule_router.route(query, available_retrievers, **kwargs)

        if decision is not None:
            logger.info(
                f"Rule-based routing: {decision.intent.value} "
                f"(confidence: {decision.confidence:.0%})"
            )
            return decision

        # Fall back to LLM routing
        if self._llm_router is not None:
            decision = self._llm_router.route(query, available_retrievers, **kwargs)
            logger.info(
                f"LLM routing: {decision.intent.value} "
                f"(confidence: {decision.confidence:.0%})"
            )
            return decision

        # No LLM router available - return default
        from agentic_search.core.models import QueryIntent

        logger.warning("No LLM router available, using default routing")
        return RoutingDecision(
            intent=QueryIntent.LOCAL_10K,
            confidence=0.5,
            reason="Default routing (no LLM router configured)",
            search_query=query,
            requires_web=False,
            keywords=[],
            target_retrievers=["qdrant_10k_data"],
        )
