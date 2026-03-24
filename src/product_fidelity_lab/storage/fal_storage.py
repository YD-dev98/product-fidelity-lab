"""Upload images to fal.ai storage with local caching to prevent re-uploads."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import fal_client
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


class FalStorage:
    """Manages image uploads to fal.ai with a persistent URL cache."""

    def __init__(self, cache_file: Path) -> None:
        self._cache_file = cache_file
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = self._load_cache()

    def _load_cache(self) -> dict[str, str]:
        if self._cache_file.exists():
            return json.loads(self._cache_file.read_text())  # type: ignore[no-any-return]
        return {}

    def _save_cache(self) -> None:
        self._cache_file.write_text(json.dumps(self._cache, indent=2))

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    async def upload_image(self, local_path: Path) -> str:
        """Upload an image to fal.ai storage, returning the URL.

        Uses a file-content-hash cache to avoid re-uploading identical files.
        """
        content_hash = self._file_hash(local_path)

        if content_hash in self._cache:
            url = self._cache[content_hash]
            logger.debug("fal_storage.cache_hit", path=str(local_path), url=url)
            return url

        url: str = await fal_client.upload_file_async(local_path)  # type: ignore[reportUnknownMemberType]
        self._cache[content_hash] = url
        self._save_cache()
        logger.info("fal_storage.uploaded", path=str(local_path), url=url)
        return url
