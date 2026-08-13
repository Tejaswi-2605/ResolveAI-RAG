"""
_bootstrap.py — make `app` importable from the repository root.

The evaluation suite lives OUTSIDE `backend/` because it evaluates the system
as a whole, not just the backend package. That means it has to put `backend/`
on the import path. Doing it in ONE place, imported first by each entry point,
keeps the `sys.path` manipulation from being copy-pasted into four files.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
