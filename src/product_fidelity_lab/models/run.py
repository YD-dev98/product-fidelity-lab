from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunType(StrEnum):
    EVALUATION = "evaluation"
    GENERATION = "generation"
    PERTURBATION = "perturbation"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.INTERRUPTED)


class LayerState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Run(BaseModel):
    id: str
    type: RunType
    status: RunStatus = RunStatus.PENDING
    layer_states: dict[str, LayerState] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    config: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    score: float | None = None
    grade: str | None = None
    passed: bool | None = None
    duration_ms: int | None = None
    cost: float | None = None


class RunArtifact(BaseModel):
    run_id: str
    artifact_type: str
    file_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
