from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.config import Settings, reset_settings


class TestSettings:
    def setup_method(self) -> None:
        reset_settings()

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAL_KEY", "fk-123")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-456")
        s = Settings()
        assert s.fal_key == "fk-123"
        assert s.gemini_api_key == "gk-456"

    def test_defaults(self) -> None:
        s = Settings()  # env vars set by conftest
        assert s.data_dir == Path(s.data_dir)
        assert s.fal_timeout_s == 120
        assert s.fal_max_concurrent == 3
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.debug is False

    def test_keys_default_empty_and_not_live_ready(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)  # no .env file here
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.fal_key == ""
        assert s.gemini_api_key == ""
        assert s.live_ready is False

    def test_live_ready_with_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAL_KEY", "fk-123")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-456")
        s = Settings()
        assert s.live_ready is True

    def test_custom_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FAL_TIMEOUT_S", "60")
        monkeypatch.setenv("FAL_MAX_CONCURRENT", "5")
        monkeypatch.setenv("PORT", "9000")
        s = Settings()
        assert s.fal_timeout_s == 60
        assert s.fal_max_concurrent == 5
        assert s.port == 9000
