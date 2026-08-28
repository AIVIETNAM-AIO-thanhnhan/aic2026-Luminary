"""AIC 2026 video retrieval toolkit."""

import os
from pathlib import Path

# faiss and torch each bundle their own OpenMP runtime; loading both in one
# process on macOS segfaults (SIGSEGV, no traceback) the moment either one
# spins up its thread pool. This has to be set before either gets imported
# anywhere, so it lives at package import time rather than near the imports
# that trigger it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# python-dotenv is a declared dependency but nothing actually called it: without
# this, GEMINI_API_KEY in .env never reaches os.environ, so query expansion
# silently takes the no-key branch straight to the rule-based fallback (no
# exception, no signal) - it looks like a working offline mode when it is
# actually just a missing load step.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a core dependency
    pass
