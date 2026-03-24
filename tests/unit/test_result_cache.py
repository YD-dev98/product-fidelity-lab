from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.storage.result_cache import ResultCache, cache_key


class TestCacheKey:
    def test_deterministic(self) -> None:
        inputs = {"model": "flux", "prompt": "a photo", "seed": 42}
        k1 = cache_key(inputs)
        k2 = cache_key(inputs)
        assert k1 == k2

    def test_order_independent(self) -> None:
        k1 = cache_key({"a": 1, "b": 2})
        k2 = cache_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        k1 = cache_key({"a": 1})
        k2 = cache_key({"a": 2})
        assert k1 != k2

    def test_sha256_hex(self) -> None:
        k = cache_key({"x": 1})
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


class TestResultCache:
    @pytest.fixture
    def cache(self, tmp_path: Path) -> ResultCache:
        return ResultCache(tmp_path / "test_cache")

    def test_miss_returns_none(self, cache: ResultCache) -> None:
        assert cache.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_or_compute_stores(self, cache: ResultCache) -> None:
        inputs = {"model": "test", "input": "data"}
        call_count = 0

        async def compute() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"output": "result"}

        # First call — computes
        entry = await cache.get_or_compute(inputs, compute)
        assert entry["result"] == {"output": "result"}
        assert call_count == 1

        # Second call — cache hit
        entry2 = await cache.get_or_compute(inputs, compute)
        assert entry2["result"] == {"output": "result"}
        assert call_count == 1  # compute not called again

    @pytest.mark.asyncio
    async def test_stores_metadata(self, cache: ResultCache) -> None:
        inputs = {"x": 1}

        async def compute() -> dict[str, int]:
            return {"v": 42}

        entry = await cache.get_or_compute(inputs, compute)
        assert "cached_at" in entry
        assert "duration_ms" in entry
        assert entry["inputs"] == inputs

    def test_clear(self, cache: ResultCache) -> None:
        key = cache_key({"a": 1})
        cache._cache.set(key, {"result": "data"})
        assert cache.get(key) is not None
        cache.clear()
        assert cache.get(key) is None

    def test_close(self, cache: ResultCache) -> None:
        cache.close()  # should not raise
