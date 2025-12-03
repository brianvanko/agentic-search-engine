#!/usr/bin/env python
"""
Streamlit entry point for the refactored agentic search engine.

Run with: streamlit run app.py
"""
import os
# Disable MPS to avoid segfault with nomic model on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

# Add src to path for local development
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agentic_search.ui.streamlit_app import main

if __name__ == "__main__":
    main()
