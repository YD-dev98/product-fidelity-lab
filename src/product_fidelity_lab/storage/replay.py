"""Demo-safe replay mode — serve precomputed runs without API calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


class ReplayStore:
    """Load and serve precomputed demo runs from disk."""

    def __init__(self, replay_dir: Path) -> None:
        self._replay_dir = replay_dir
        self._runs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._replay_dir.exists():
            logger.info("replay.no_data", path=str(self._replay_dir))
            return

        for run_dir in sorted(self._replay_dir.iterdir()):
            run_file = run_dir / "run.json"
            if run_file.exists():
                data: dict[str, Any] = json.loads(run_file.read_text())
                run_id = data.get("id", run_dir.name)
                self._runs[run_id] = data
                logger.debug("replay.loaded", run_id=run_id)

        logger.info("replay.ready", count=len(self._runs))

    def list_runs(self) -> list[dict[str, Any]]:
        """List all precomputed replay runs."""
        return list(self._runs.values())

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific replay run by ID."""
        return self._runs.get(run_id)

    @property
    def available(self) -> bool:
        return len(self._runs) > 0
