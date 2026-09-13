"""Simple, deterministic on-disk JSON cache for LLM and VLM calls.

Every call is content-addressed by a SHA-256 hash of its canonical JSON key,
so re-runs with identical inputs produce bit-identical results at zero API
cost.  This is critical for reproducibility under evaluation conditions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """Caches JSON-serializable values keyed by a JSON-serializable dict."""

    def __init__(self, cache_dir: Path, namespace: str) -> None:
        self._dir = cache_dir / namespace
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: dict) -> Path:
        canonical = json.dumps(key, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.json"

    def get(self, key: dict) -> Optional[Any]:
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: dict, value: Any) -> None:
        path = self._path_for(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
