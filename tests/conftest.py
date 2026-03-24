from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set required env vars for tests so Settings can be instantiated."""
    monkeypatch.setenv("FAL_KEY", "test-fal-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "data" / "test.db"))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a minimal valid JPEG for tests."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "sample.jpg"
    img.save(path)
    return path
