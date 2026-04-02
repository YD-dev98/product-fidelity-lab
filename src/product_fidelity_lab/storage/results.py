"""Structured results storage for render and evaluation runs.

Every run gets a timestamped directory under data/results/:

    data/results/
        2026-04-02T03-11-48_adapter-comparison/
            manifest.json          # run metadata, scores, config
            reference_only/
                seed200.jpg
                seed201.jpg
                repaired.png
            adapter/
                seed200.jpg
                seed201.jpg
                repaired.png
        2026-04-02T14-30-00_smoke-test/
            manifest.json
            candidates/
                seed42.jpg
                seed43.jpg
            repaired.png

Images are downloaded eagerly so they survive CDN expiry.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

RESULTS_ROOT = Path("data/results")


class ResultsWriter:
    """Write structured results to a timestamped directory."""

    def __init__(self, label: str, root: Path | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        safe_label = label.replace(" ", "-").replace("/", "-").lower()
        self._dir = (root or RESULTS_ROOT) / f"{ts}_{safe_label}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {
            "label": label,
            "created_at": datetime.now().isoformat(),
            "images": {},
        }

    @property
    def dir(self) -> Path:
        return self._dir

    async def save_image(
        self,
        url: str,
        name: str,
        *,
        subdir: str | None = None,
    ) -> Path:
        """Download an image from URL and save it locally.

        Args:
            url: Remote image URL (fal CDN, etc.)
            name: Filename without extension (e.g. "seed200")
            subdir: Optional subdirectory (e.g. "reference_only", "adapter")

        Returns:
            Local path to the saved file.
        """
        from product_fidelity_lab.product.ingest import _detect_mime

        # Support both remote URLs and local file:// paths
        if url.startswith("file://"):
            local = Path(url.removeprefix("file://"))
            img_bytes = local.read_bytes()
        else:
            from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

            img_bytes = await fetch_image_bytes(url)

        mime = _detect_mime(url)
        ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime, ".jpg")

        target_dir = self._dir / subdir if subdir else self._dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{name}{ext}"
        path.write_bytes(img_bytes)

        # Track in manifest
        key = f"{subdir}/{name}{ext}" if subdir else f"{name}{ext}"
        self._manifest["images"][key] = {
            "source_url": url,
            "local_path": str(path),
            "size_bytes": len(img_bytes),
        }

        logger.debug("results.saved_image", path=str(path), size=len(img_bytes))
        return path

    def set_metadata(self, key: str, value: Any) -> None:
        """Add arbitrary metadata to the manifest."""
        self._manifest[key] = value

    def write_manifest(self) -> Path:
        """Write the manifest.json file. Call this when done."""
        manifest_path = self._dir / "manifest.json"
        manifest_path.write_text(json.dumps(self._manifest, indent=2, default=str))
        logger.info("results.manifest_written", path=str(manifest_path))
        return manifest_path
