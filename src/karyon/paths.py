"""Cache directory resolution for karyon datasets."""
from __future__ import annotations
import os
from pathlib import Path


def cache_dir() -> Path:
    """Where fetched datasets are cached. Override with $KARYON_CACHE."""
    base = os.environ.get("KARYON_CACHE")
    d = Path(base) if base else Path.home() / ".cache" / "karyon"
    d.mkdir(parents=True, exist_ok=True)
    return d
