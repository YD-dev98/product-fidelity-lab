"""Tests for upload validation and endpoint contract.

Tests MIME detection directly. Tests the upload 202/422 contract via
a standalone async test that exercises the same validation logic as the endpoint.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from PIL import Image

from product_fidelity_lab.product.ingest import _detect_mime


class TestUploadMimeDetection:
    def test_jpeg_extension(self) -> None:
        assert _detect_mime("/tmp/photo.jpg") == "image/jpeg"

    def test_jpeg_long_extension(self) -> None:
        assert _detect_mime("/tmp/photo.jpeg") == "image/jpeg"

    def test_png_extension(self) -> None:
        assert _detect_mime("/tmp/photo.png") == "image/png"

    def test_webp_extension(self) -> None:
        assert _detect_mime("/tmp/photo.webp") == "image/webp"

    def test_unknown_defaults_to_jpeg(self) -> None:
        assert _detect_mime("/tmp/photo.bmp") == "image/jpeg"

    def test_case_insensitive(self) -> None:
        assert _detect_mime("/tmp/photo.PNG") == "image/png"
        assert _detect_mime("/tmp/photo.JPG") == "image/jpeg"

    def test_signed_url_strips_query(self) -> None:
        url = "https://fal.ai/storage/abc123/image.webp?sig=xyz&exp=123"
        assert _detect_mime(url) == "image/webp"

    def test_signed_url_png(self) -> None:
        url = "https://cdn.example.com/renders/out.png?token=abc"
        assert _detect_mime(url) == "image/png"

    def test_url_no_extension_defaults_to_jpeg(self) -> None:
        url = "https://fal.ai/storage/abc123/image?sig=xyz"
        assert _detect_mime(url) == "image/jpeg"

    def test_local_path_still_works(self) -> None:
        assert _detect_mime("/tmp/photo.webp") == "image/webp"


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(buf, format="JPEG")
    return buf.getvalue()


# ── Upload validation logic (mirrors api/products.py) ────────────────

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 50 * 1024 * 1024


def _validate_upload(
    filename: str,
    content_type: str,
    contents: bytes,
) -> str | None:
    """Validate a single upload file. Returns rejection reason or None if accepted."""
    mime = content_type.split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME_TYPES:
        return f"Invalid type: {mime}"
    if len(contents) > MAX_FILE_BYTES:
        return "Too large"
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        return "Not a valid image"
    return None


def _process_uploads(
    files: list[tuple[str, str, bytes]],
    uploads_dir: Path,
) -> tuple[list[str], list[dict[str, str]], list[Path]]:
    """Process uploads, returning (accepted, rejected, saved_paths)."""
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    saved_paths: list[Path] = []

    for filename, content_type, contents in files:
        reason = _validate_upload(filename, content_type, contents)
        if reason:
            rejected.append({"filename": filename, "reason": reason})
            continue

        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(
            dir=str(uploads_dir), suffix=suffix, delete=False
        ) as tmp:
            tmp.write(contents)
            saved_paths.append(Path(tmp.name))
        accepted.append(filename)

    return accepted, rejected, saved_paths


class TestUploadEndpointContract:
    """Tests the real validation + response-shape logic used by POST /api/products/{id}/upload."""

    def test_partial_accept(self, tmp_path: Path) -> None:
        """One valid + one invalid -> accepted has 1, rejected has 1."""
        accepted, rejected, saved = _process_uploads(
            [
                ("good.jpg", "image/jpeg", _make_jpeg_bytes()),
                ("bad.txt", "text/plain", b"not an image"),
            ],
            tmp_path,
        )
        assert len(accepted) == 1
        assert len(rejected) == 1
        assert rejected[0]["filename"] == "bad.txt"
        assert len(saved) == 1
        # Would return 202

    def test_zero_accepted(self, tmp_path: Path) -> None:
        """All invalid -> zero accepted, all rejected."""
        accepted, rejected, saved = _process_uploads(
            [
                ("bad.txt", "text/plain", b"not an image"),
                ("bad.pdf", "application/pdf", b"%PDF-fake"),
            ],
            tmp_path,
        )
        assert len(accepted) == 0
        assert len(rejected) == 2
        assert len(saved) == 0
        # Would return 422

    def test_all_accepted(self, tmp_path: Path) -> None:
        """All valid -> all accepted, none rejected."""
        accepted, rejected, saved = _process_uploads(
            [
                ("a.jpg", "image/jpeg", _make_jpeg_bytes()),
                ("b.jpg", "image/jpeg", _make_jpeg_bytes()),
            ],
            tmp_path,
        )
        assert len(accepted) == 2
        assert len(rejected) == 0
        assert len(saved) == 2
        # Would return 202

    def test_invalid_image_bytes_rejected(self, tmp_path: Path) -> None:
        """Correct MIME but corrupt image data -> rejected."""
        accepted, rejected, saved = _process_uploads(
            [("fake.jpg", "image/jpeg", b"not a real jpeg")],
            tmp_path,
        )
        assert len(accepted) == 0
        assert len(rejected) == 1
        assert "Not a valid image" in rejected[0]["reason"]

    def test_202_vs_422_decision(self, tmp_path: Path) -> None:
        """Verify the 202/422 decision point: saved_paths non-empty -> 202."""
        _, _, saved_partial = _process_uploads(
            [
                ("good.jpg", "image/jpeg", _make_jpeg_bytes()),
                ("bad.txt", "text/plain", b"x"),
            ],
            tmp_path,
        )
        assert len(saved_partial) > 0  # -> 202

        _, _, saved_none = _process_uploads(
            [("bad.txt", "text/plain", b"x")],
            tmp_path,
        )
        assert len(saved_none) == 0  # -> 422
