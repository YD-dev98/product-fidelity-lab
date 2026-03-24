"""Typed async wrapper around fal_client with retries, timeout, and concurrency control."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import fal_client
import structlog

logger = structlog.get_logger()

# HTTP status codes that trigger retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class FalClientError(Exception):
    """Error from the fal.ai API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FalClient:
    """Async fal.ai client with retries, timeout, and concurrency limiting."""

    def __init__(
        self,
        *,
        timeout_s: int = 120,
        max_concurrent: int = 3,
        max_retries: int = 3,
    ) -> None:
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def subscribe(
        self,
        model_id: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit a request to a fal.ai model and wait for the result.

        Retries on 429/5xx with exponential backoff. Respects concurrency limit.
        """
        async with self._semaphore:
            return await self._subscribe_with_retry(model_id, input_data)

    async def _subscribe_with_retry(
        self,
        model_id: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            start = time.monotonic()
            log = logger.bind(model_id=model_id, attempt=attempt)

            try:
                result: Any = await asyncio.wait_for(
                    fal_client.subscribe_async(
                        model_id,
                        arguments=input_data,
                    ),
                    timeout=self._timeout_s,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info("fal.success", duration_ms=duration_ms)
                return result  # type: ignore[no-any-return]

            except TimeoutError:
                duration_ms = int((time.monotonic() - start) * 1000)
                log.warning("fal.timeout", duration_ms=duration_ms)
                last_error = FalClientError(
                    f"Timeout after {self._timeout_s}s for {model_id}"
                )

            except Exception as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                status_code = getattr(exc, "status_code", None)

                if status_code and status_code not in _RETRYABLE_STATUSES:
                    log.error("fal.non_retryable", status_code=status_code, error=str(exc))
                    raise FalClientError(str(exc), status_code=status_code) from exc

                log.warning(
                    "fal.retryable_error",
                    status_code=status_code,
                    error=str(exc),
                    duration_ms=duration_ms,
                )
                last_error = exc

            if attempt < self._max_retries:
                backoff = 2 ** (attempt - 1)
                log.info("fal.backoff", seconds=backoff)
                await asyncio.sleep(backoff)

        raise FalClientError(
            f"Failed after {self._max_retries} attempts for {model_id}: {last_error}"
        )

    async def upload_file(self, path: str) -> str:
        """Upload a file to fal.ai storage and return the URL."""
        from pathlib import Path as _Path

        url: str = await fal_client.upload_file_async(_Path(path))  # type: ignore[reportUnknownMemberType]
        logger.info("fal.uploaded", path=path, url=url)
        return url
