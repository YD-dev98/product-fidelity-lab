"""Benchmark protocol enforcement for generation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from product_fidelity_lab.models.golden_spec import GoldenSpec


class BenchmarkProtocol(BaseModel):
    """Defines what references are allowed/forbidden for a benchmark."""

    benchmark_id: str
    mode: str  # "held_out_synthesis" | "repair"
    target_shot_id: str
    description: str = ""
    allowed_reference_shot_ids: list[str] = Field(default_factory=list)
    forbidden_shot_ids: list[str] = Field(default_factory=list)

    def validate_references(self, shot_ids: list[str]) -> None:
        """Raise ValueError if any referenced shot violates the protocol."""
        for sid in shot_ids:
            if sid in self.forbidden_shot_ids:
                raise ValueError(
                    f"Protocol {self.benchmark_id!r} forbids "
                    f"reference to {sid!r}"
                )
            if (
                self.allowed_reference_shot_ids
                and sid not in self.allowed_reference_shot_ids
            ):
                raise ValueError(
                    f"Protocol {self.benchmark_id!r} only allows "
                    f"references from {self.allowed_reference_shot_ids}, "
                    f"got {sid!r}"
                )

    def select_references(
        self,
        all_specs: list[GoldenSpec],
        shot_ids: list[str] | None = None,
        max_refs: int = 4,
    ) -> list[str]:
        """Select reference URLs respecting the protocol.

        Args:
            all_specs: All available golden specs.
            shot_ids: Specific shots to use. If None, uses allowed list.
            max_refs: Maximum number of references to return.
        """
        ids = shot_ids or self.allowed_reference_shot_ids
        self.validate_references(ids)

        urls: list[str] = []
        for sid in ids:
            if len(urls) >= max_refs:
                break
            spec = next(
                (s for s in all_specs if s.shot_id == sid), None,
            )
            if spec and spec.image_url:
                urls.append(spec.image_url)
        return urls


def load_protocol(path: Path) -> BenchmarkProtocol:
    """Load a benchmark protocol from a JSON file."""
    data = json.loads(path.read_text())
    return BenchmarkProtocol.model_validate(data)
