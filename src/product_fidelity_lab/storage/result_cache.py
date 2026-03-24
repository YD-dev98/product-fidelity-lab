"""Deterministic cache for paid API outputs using diskcache."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import diskcache  # type: ignore[reportMissingTypeStubs]
import structlog

logger = structlog.get_logger()


def cache_key(inputs: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 cache key from sorted, serialized inputs."""
    serialized = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


class ResultCache:
    """Per-directory deterministic cache for API responses.

    Each cache directory is independent so caches can be selectively cleared.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached result by key."""
        result: Any = self._cache.get(key)  # type: ignore[reportUnknownMemberType]
        if result is not None:
            logger.debug("cache.hit", key=key[:16], cache_dir=self._cache_dir.name)
            return result  # type: ignore[no-any-return]
        logger.debug("cache.miss", key=key[:16], cache_dir=self._cache_dir.name)
        return None

    async def get_or_compute(
        self,
        inputs: dict[str, Any],
        compute_fn: Any,
    ) -> dict[str, Any]:
        """Return cached result or call compute_fn, cache, and return.

        Args:
            inputs: Dict used to compute the cache key.
            compute_fn: Async callable that produces the result on cache miss.

        Returns:
            The cached or freshly computed result dict.
        """
        key = cache_key(inputs)
        cached = self.get(key)
        if cached is not None:
            return cached

        start = time.monotonic()
        result: dict[str, Any] = await compute_fn()
        duration_ms = int((time.monotonic() - start) * 1000)

        entry: dict[str, Any] = {
            "result": result,
            "cached_at": time.time(),
            "duration_ms": duration_ms,
            "inputs": inputs,
        }
        self._cache.set(key, entry)  # type: ignore[reportUnknownMemberType]
        logger.info(
            "cache.store",
            key=key[:16],
            duration_ms=duration_ms,
            cache_dir=self._cache_dir.name,
        )
        return entry

    def clear(self) -> None:
        """Clear all entries in this cache."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying cache."""
        self._cache.close()
