"""Streamlit web interface for the agentic search engine."""

import time
from typing import Dict, Any

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Agentic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_pipeline():
    """Initialize the RAG pipeline (cached in session state)."""
    if "pipeline" not in st.session_state:
        with st.spinner("Initializing search engine..."):
            from agentic_search.config import create_pipeline
            st.session_state.pipeline = create_pipeline(
                enable_cache=True,
                enable_web_search=True,
            )
    return st.session_state.pipeline


def init_session_state():
    """Initialize session state variables."""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def display_cache_stats(stats: Dict[str, Any]):
    """Display cache statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", stats.get("total_queries", 0))
    with col2:
        st.metric("Cache Hits", stats.get("cache_hits", 0))
    with col3:
        st.metric("Hit Rate", f"{stats.get('hit_rate_percent', 0):.1f}%")
    with col4:
        time_saved = stats.get("total_time_saved_ms", 0) / 1000
        st.metric("Time Saved", f"{time_saved:.1f}s")


def display_sources(sources: list):
    """Display source documents."""
    if not sources:
        st.info("No source documents used.")
        return

    st.subheader(f"📚 Sources ({len(sources)} documents)")

    for i, source in enumerate(sources, 1):
        source_type = source.get("source", "unknown")
        score = source.get("score", 0)
        content = source.get("content", "")[:500]
        metadata = source.get("metadata", {})

        icon = "📄" if source_type == "10k_data" else "🌐" if source_type == "web_search" else "📖"

        with st.expander(f"{icon} Source {i} - {source_type} (Score: {score:.2f})"):
            st.markdown(content)
            if metadata:
                st.json(metadata)


def main():
    """Main Streamlit application."""
    init_session_state()

    # Header
    st.title("🔍 Agentic Search Engine")
    st.markdown("""
    A RAG-powered search engine with semantic caching and intelligent query routing.
    Search across **10-K financial filings** and get answers powered by AI.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        allow_web_search = st.toggle(
            "Allow Web Search",
            value=True,
            help="Enable DuckDuckGo search for current events",
        )

        top_k = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=5,
            help="How many documents to retrieve",
        )

        st.session_state.show_debug = st.toggle(
            "Show Debug Info",
            value=st.session_state.show_debug,
            help="Display routing decisions and timing",
        )

        st.divider()

        # Cache controls
        st.header("📊 Cache Statistics")

        try:
            pipeline = init_pipeline()
            stats = pipeline.get_cache_stats()
            display_cache_stats(stats)

            if st.button("🗑️ Clear Cache"):
                pipeline.clear_cache()
                st.success("Cache cleared!")
                st.rerun()

        except Exception as e:
            st.error(f"Pipeline not initialized: {e}")

        st.divider()

        # Retriever info
        st.header("📁 Available Retrievers")
        try:
            pipeline = init_pipeline()
            info = pipeline.get_retriever_info()
            for r in info:
                st.markdown(f"**{r.get('name', 'unknown')}**: {r.get('source_type', 'unknown')}")
        except Exception as e:
            st.info(f"Retriever info unavailable: {e}")

    # Main search interface
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., What were Apple's revenue trends in the last fiscal year?",
            label_visibility="collapsed",
        )

    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    # Example queries
    st.markdown("**Try these examples:**")
    example_cols = st.columns(3)

    examples = [
        "What are the main risk factors mentioned in the 10-K filings?",
        "How do I build an AI agent with OpenAI?",
        "Latest Nvidia earnings news",
    ]

    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                query = example
                search_clicked = True

    # Process search
    if search_clicked and query:
        try:
            pipeline = init_pipeline()

            with st.status("Searching...", expanded=True) as status:
                st.write("Checking cache...")
                time.sleep(0.1)

                st.write("Routing query...")
                result = pipeline.search(
                    query=query,
                    top_k=top_k,
                    use_cache=True,
                )

                status.update(label="Search complete!", state="complete")

            # Add to history
            st.session_state.search_history.insert(0, result.to_dict())
            st.session_state.search_history = st.session_state.search_history[:50]

            # Display results
            st.divider()

            # Cache hit indicator
            if result.cache_hit:
                st.success("⚡ **Cache Hit!** Response retrieved from semantic cache.")
                if result.cache_metadata:
                    sim = result.cache_metadata.get("similarity_score", 0)
                    st.caption(
                        f"Similarity: {sim:.2%} | "
                        f"Cached question: {result.cache_metadata.get('cached_question', 'N/A')[:100]}..."
                    )
            else:
                st.info("🔄 **Cache Miss** - Performed full RAG search")

            # Response
            st.subheader("💬 Answer")
            st.markdown(result.response)

            # Debug info
            if st.session_state.show_debug:
                st.divider()
                debug_col1, debug_col2 = st.columns(2)

                with debug_col1:
                    st.subheader("🛣️ Routing Decision")
                    routing = result.routing_decision
                    st.markdown(f"""
                    - **Intent**: `{routing.get('intent', 'N/A')}`
                    - **Confidence**: {routing.get('confidence', 0):.0%}
                    - **Reason**: {routing.get('reason', 'N/A')}
                    - **Web Search**: {'Yes' if routing.get('requires_web') else 'No'}
                    """)

                with debug_col2:
                    st.subheader("⏱️ Timing")
                    st.markdown(f"""
                    - **Total**: {result.timing.get('total_ms', 0):.0f}ms
                    - **Cache Lookup**: {result.timing.get('cache_lookup_ms', 0):.0f}ms
                    - **Routing**: {result.timing.get('routing_ms', 0):.0f}ms
                    - **Retrieval**: {result.timing.get('retrieval_ms', 0):.0f}ms
                    - **Generation**: {result.timing.get('generation_ms', 0):.0f}ms
                    """)

            # Sources
            st.divider()
            display_sources(result.sources)

        except FileNotFoundError as e:
            st.error(f"❌ Vector store not found: {e}")
            st.info("Please ensure the Qdrant data is available at the configured path.")
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            import traceback
            if st.session_state.show_debug:
                st.code(traceback.format_exc())

    # Search history
    if st.session_state.search_history:
        st.divider()
        st.subheader("📜 Recent Searches")

        for i, item in enumerate(st.session_state.search_history[:5]):
            with st.expander(f"Q: {item['query'][:80]}...", expanded=False):
                st.markdown(f"**Answer:** {item['response'][:300]}...")
                st.caption(
                    f"Cache hit: {item['cache_hit']} | "
                    f"Time: {item['timing'].get('total_ms', 0):.0f}ms"
                )


if __name__ == "__main__":
    main()
