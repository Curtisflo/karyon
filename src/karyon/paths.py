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


def network_allowed() -> bool:
    """False when ``KARYON_NO_NETWORK`` is set in the environment.

    Data loaders consult this before hitting the network. When disabled they
    raise their usual ``DatasetUnavailable`` instead of fetching, so already
    cached datasets still load while offline and CI runs stay fast and
    deterministic (the online portions of the test suite skip cleanly).
    """
    return not os.environ.get("KARYON_NO_NETWORK")
