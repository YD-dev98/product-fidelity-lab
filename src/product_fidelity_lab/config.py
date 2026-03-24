from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    fal_key: str = Field(default="", description="fal.ai API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model ID",
    )

    replay_mode: bool = Field(
        default=False,
        description="Replay-only mode",
        validation_alias=AliasChoices("replay_mode", "PFL_REPLAY_MODE"),
    )

    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    db_path: Path = Field(default=Path("data/pfl.db"), description="SQLite database path")

    fal_timeout_s: int = Field(default=120, description="fal.ai request timeout in seconds")
    fal_max_concurrent: int = Field(default=3, description="Max concurrent fal.ai requests")

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    @property
    def live_ready(self) -> bool:
        """Whether API keys are configured for live evaluation."""
        return bool(self.fal_key) and bool(self.gemini_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()  # type: ignore[reportCallIssue]
    return _settings


def reset_settings() -> None:
    """Reset cached settings. Used in tests."""
    global _settings  # noqa: PLW0603
    _settings = None
