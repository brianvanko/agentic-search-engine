#!/usr/bin/env python
"""
CLI entry point for the refactored agentic search engine.

Run with: python main_v2.py "your query"
"""
import sys
from pathlib import Path

# Add src to path for local development
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agentic_search.ui.cli import main

if __name__ == "__main__":
    main()
