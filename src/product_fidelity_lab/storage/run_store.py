"""SQLite-backed run metadata store with artifact folder management."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import aiosqlite
import structlog

from product_fidelity_lab.models.run import (
    LayerState,
    Run,
    RunArtifact,
    RunStatus,
    RunType,
)

logger = structlog.get_logger()

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    layer_states_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    score REAL,
    grade TEXT,
    passed INTEGER,
    duration_ms INTEGER,
    cost REAL
)
"""

CREATE_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS run_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
)
"""

CREATE_ARTIFACTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON run_artifacts(run_id)
"""


class RunStore:
    """Manages run metadata in SQLite and artifact files on disk."""

    def __init__(self, db_path: Path, runs_dir: Path) -> None:
        self._db_path = db_path
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(CREATE_RUNS_TABLE)
            await db.execute(CREATE_ARTIFACTS_TABLE)
            await db.execute(CREATE_ARTIFACTS_INDEX)
            await db.commit()
        logger.info("run_store.initialized", db_path=str(self._db_path))

    async def recover_interrupted(self) -> int:
        """Mark non-terminal runs as interrupted on startup. Returns count."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT id, layer_states_json FROM runs WHERE status NOT IN (?, ?, ?)",
                (RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.INTERRUPTED),
            )
            rows = await cursor.fetchall()
            count = 0
            for row in rows:
                run_id: str = row[0]
                layer_states: dict[str, str] = json.loads(row[1])
                # Normalize layer states
                for layer_key, state in layer_states.items():
                    if state in (LayerState.RUNNING, LayerState.PENDING):
                        layer_states[layer_key] = LayerState.INTERRUPTED
                sql = (
                    "UPDATE runs SET status = ?, layer_states_json = ?,"
                    " updated_at = ? WHERE id = ?"
                )
                await db.execute(
                    sql,
                    (RunStatus.INTERRUPTED, json.dumps(layer_states), _now_iso(), run_id),
                )
                count += 1
            await db.commit()
        if count:
            logger.warning("run_store.recovered_interrupted", count=count)
        return count

    async def create_run(
        self,
        run_type: RunType,
        config: dict[str, Any] | None = None,
    ) -> Run:
        """Create a new run and its artifact directory."""
        run_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        run = Run(
            id=run_id,
            type=run_type,
            status=RunStatus.PENDING,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            config=config or {},
        )
        artifact_dir = self._runs_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """INSERT INTO runs (id, type, status, layer_states_json,
                   created_at, updated_at, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run.type,
                    run.status,
                    json.dumps({}),
                    now,
                    now,
                    json.dumps(run.config),
                ),
            )
            await db.commit()
        logger.info("run_store.created", run_id=run_id, type=run_type)
        return run

    async def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_run(row)

    async def list_runs(
        self,
        type_filter: RunType | None = None,
        limit: int = 50,
    ) -> list[Run]:
        """List runs, optionally filtered by type."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            if type_filter:
                cursor = await db.execute(
                    "SELECT * FROM runs WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                    (type_filter, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [_row_to_run(row) for row in rows]

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        layer_states: dict[str, LayerState] | None = None,
    ) -> None:
        """Update run status and optionally layer states."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            if layer_states is not None:
                sql = (
                    "UPDATE runs SET status = ?, layer_states_json = ?,"
                    " updated_at = ? WHERE id = ?"
                )
                await db.execute(sql, (status, json.dumps(layer_states), _now_iso(), run_id))
            else:
                await db.execute(
                    "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, _now_iso(), run_id),
                )
            await db.commit()

    async def update_result(
        self,
        run_id: str,
        *,
        result: dict[str, Any] | None = None,
        score: float | None = None,
        grade: str | None = None,
        passed: bool | None = None,
        duration_ms: int | None = None,
        cost: float | None = None,
    ) -> None:
        """Update run results."""
        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now_iso()]

        if result is not None:
            updates.append("result_json = ?")
            params.append(json.dumps(result))
        if score is not None:
            updates.append("score = ?")
            params.append(score)
        if grade is not None:
            updates.append("grade = ?")
            params.append(grade)
        if passed is not None:
            updates.append("passed = ?")
            params.append(int(passed))
        if duration_ms is not None:
            updates.append("duration_ms = ?")
            params.append(duration_ms)
        if cost is not None:
            updates.append("cost = ?")
            params.append(cost)

        params.append(run_id)
        sql = f"UPDATE runs SET {', '.join(updates)} WHERE id = ?"
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(sql, params)
            await db.commit()

    async def add_artifact(
        self,
        run_id: str,
        artifact_type: str,
        file_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunArtifact:
        """Record an artifact for a run."""
        artifact = RunArtifact(
            run_id=run_id,
            artifact_type=artifact_type,
            file_path=file_path,
            metadata=metadata or {},
        )
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                """INSERT INTO run_artifacts (run_id, artifact_type, file_path, metadata_json)
                   VALUES (?, ?, ?, ?)""",
                (run_id, artifact_type, file_path, json.dumps(artifact.metadata)),
            )
            await db.commit()
        return artifact

    async def get_artifacts(self, run_id: str) -> list[RunArtifact]:
        """Get all artifacts for a run."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM run_artifacts WHERE run_id = ?", (run_id,)
            )
            rows = await cursor.fetchall()
            return [
                RunArtifact(
                    run_id=row["run_id"],
                    artifact_type=row["artifact_type"],
                    file_path=row["file_path"],
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in rows
            ]

    def artifact_dir(self, run_id: str) -> Path:
        """Return the artifact directory for a run."""
        d = self._runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def delete_run(self, run_id: str) -> None:
        """Delete a run and its artifacts."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("DELETE FROM run_artifacts WHERE run_id = ?", (run_id,))
            await db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            await db.commit()
        artifact_dir = self._runs_dir / run_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _row_to_run(row: Any) -> Run:
    return Run(
        id=row["id"],
        type=RunType(row["type"]),
        status=RunStatus(row["status"]),
        layer_states={
            k: LayerState(v) for k, v in json.loads(row["layer_states_json"]).items()
        },
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        config=json.loads(row["config_json"]),
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        score=row["score"],
        grade=row["grade"],
        passed=bool(row["passed"]) if row["passed"] is not None else None,
        duration_ms=row["duration_ms"],
        cost=row["cost"],
    )
