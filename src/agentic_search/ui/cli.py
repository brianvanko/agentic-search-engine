"""Command-line interface for the agentic search engine."""

import argparse
import json
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Search Engine - RAG-powered search with caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are Lyft's risk factors?"
  %(prog)s -i                          # Interactive mode
  %(prog)s -w "Latest Nvidia earnings" # With web search
  %(prog)s --stats                     # Show cache statistics
  %(prog)s --collections               # List vector collections
        """,
    )

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("-w", "--web", action="store_true", help="Enable web search")
    parser.add_argument("--no-web", action="store_true", help="Disable web search")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of documents")
    parser.add_argument("--no-cache", action="store_true", help="Disable semantic cache")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show routing and timing")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache")
    parser.add_argument("--collections", action="store_true", help="List vector collections")

    args = parser.parse_args()

    # Lazy import to avoid slow startup for help text
    from agentic_search.config import create_pipeline, Settings
    from agentic_search.core.exceptions import ConfigurationError

    # Determine web search setting
    enable_web = True
    if args.no_web:
        enable_web = False
    elif args.web:
        enable_web = True

    try:
        # Create pipeline
        pipeline = create_pipeline(
            enable_cache=not args.no_cache,
            enable_web_search=enable_web,
        )

        # Handle utility commands
        if args.stats:
            stats = pipeline.get_cache_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("\n=== Cache Statistics ===")
                print(f"Total Queries: {stats.get('total_queries', 0)}")
                print(f"Cache Hits:    {stats.get('cache_hits', 0)}")
                print(f"Hit Rate:      {stats.get('hit_rate_percent', 0):.1f}%")
                print(f"Time Saved:    {stats.get('total_time_saved_ms', 0)/1000:.1f}s")
                print(f"Cache Size:    {stats.get('cache_size', 0)} entries")
            return

        if args.clear_cache:
            pipeline.clear_cache()
            print("Cache cleared successfully")
            return

        if args.collections:
            info = pipeline.get_retriever_info()
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print("\n=== Available Retrievers ===")
                for r in info:
                    print(f"- {r.get('name', 'unknown')}: {r.get('source_type', 'unknown')}")
            return

        # Handle search modes
        if args.interactive:
            print("\n=== Agentic Search Engine (Interactive Mode) ===")
            print("Type 'quit' or 'exit' to stop\n")

            while True:
                try:
                    query = input("Query: ").strip()
                    if query.lower() in ("quit", "exit", "q"):
                        print("Goodbye!")
                        break
                    if not query:
                        continue

                    result = pipeline.search(query, top_k=args.top_k)
                    _print_result(result, args.verbose, args.json)

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

        elif args.query:
            result = pipeline.search(args.query, top_k=args.top_k)
            _print_result(result, args.verbose, args.json)

        else:
            parser.print_help()

    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _print_result(result, verbose: bool, as_json: bool):
    """Print search result."""
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\n{'='*60}")

    if result.cache_hit:
        print("⚡ CACHE HIT")
        if result.cache_metadata:
            sim = result.cache_metadata.get("similarity_score", 0)
            print(f"   Similarity: {sim:.2%}")

    print(f"\n{result.response}\n")

    if verbose:
        print("-" * 60)
        routing = result.routing_decision
        print(f"Routing: {routing.get('intent', 'N/A')} "
              f"(confidence: {routing.get('confidence', 0):.0%})")
        print(f"Reason: {routing.get('reason', 'N/A')}")

        timing = result.timing
        print(f"\nTiming:")
        print(f"  Total:      {timing.get('total_ms', 0):.0f}ms")
        print(f"  Cache:      {timing.get('cache_lookup_ms', 0):.0f}ms")
        print(f"  Routing:    {timing.get('routing_ms', 0):.0f}ms")
        print(f"  Retrieval:  {timing.get('retrieval_ms', 0):.0f}ms")
        print(f"  Generation: {timing.get('generation_ms', 0):.0f}ms")

    if result.sources:
        print(f"\nSources: {len(result.sources)} documents")

    print("=" * 60)


if __name__ == "__main__":
    main()
