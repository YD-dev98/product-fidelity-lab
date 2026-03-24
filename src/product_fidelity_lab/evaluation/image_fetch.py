"""Shared image download with in-memory caching, validation, and retry."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger()

_cache: dict[str, bytes] = {}

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_CONTENT_TYPES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif",
})


class ImageFetchError(Exception):
    """Raised when image fetch fails validation."""


def _validate_url(url: str) -> None:
    """Block private/internal IPs to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageFetchError(f"Unsupported scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ImageFetchError("Missing hostname")

    # Resolve hostname and check for private IPs
    try:
        for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                raise ImageFetchError(f"URL resolves to non-public IP: {addr}")
    except socket.gaierror as exc:
        raise ImageFetchError(f"Cannot resolve hostname: {hostname}") from exc


def preload_local(url: str, local_path: Path) -> None:
    """Pre-load a local file into the cache keyed by its remote URL."""
    if url not in _cache and local_path.exists():
        _cache[url] = local_path.read_bytes()
        logger.debug("image_fetch.preloaded_local", url=url[:60])


async def fetch_image_bytes(url: str, max_retries: int = 3) -> bytes:
    """Download image bytes from a URL, with validation, caching, and retry."""
    if url in _cache:
        logger.debug("image_fetch.cache_hit", url=url[:60])
        return _cache[url]

    _validate_url(url)

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()

            # Validate content type
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                raise ImageFetchError(f"Invalid content type: {content_type}")

            # Validate size
            if len(resp.content) > MAX_IMAGE_BYTES:
                raise ImageFetchError(
                    f"Image too large: {len(resp.content)} bytes (max {MAX_IMAGE_BYTES})"
                )

            _cache[url] = resp.content
            logger.debug(
                "image_fetch.downloaded", url=url[:60], size=len(resp.content),
            )
            return resp.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("image_fetch.rate_limited", attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
            else:
                raise

    msg = f"Failed to fetch image after {max_retries} attempts: {url}"
    raise RuntimeError(msg)


def clear_cache() -> None:
    """Clear the image download cache."""
    _cache.clear()
