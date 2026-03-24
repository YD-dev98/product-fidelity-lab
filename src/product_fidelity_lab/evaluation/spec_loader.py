"""Load and validate golden specs from disk."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from product_fidelity_lab.models.golden_spec import GoldenSpec

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()


def load_spec(spec_path: Path) -> GoldenSpec:
    """Load a single golden spec from a JSON file."""
    data = json.loads(spec_path.read_text())
    return GoldenSpec.model_validate(data)


def load_all_specs(specs_dir: Path) -> list[GoldenSpec]:
    """Load all golden specs from a directory."""
    specs: list[GoldenSpec] = []
    for path in sorted(specs_dir.glob("*.json")):
        spec = load_spec(path)
        specs.append(spec)
        logger.debug("spec.loaded", shot_id=spec.shot_id, path=str(path))
    logger.info("specs.loaded", count=len(specs))
    return specs


def validate_specs(specs: list[GoldenSpec], golden_dir: Path) -> list[str]:
    """Validate that all referenced images exist. Returns list of errors."""
    errors: list[str] = []
    for spec in specs:
        image_path = golden_dir / spec.image_path
        if not image_path.exists():
            errors.append(f"Image not found for {spec.shot_id}: {spec.image_path}")
    return errors
