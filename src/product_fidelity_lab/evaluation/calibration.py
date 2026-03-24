"""Calibration: compute thresholds from training data and validate on held-out set."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class LayerDistribution(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    n: int


class CalibrationReport(BaseModel):
    training_distributions: dict[str, LayerDistribution]
    frozen_thresholds: dict[str, float]
    validation_scores: dict[str, list[float]] = Field(default_factory=dict)
    perturbation_detection: dict[str, dict[str, Any]] = Field(default_factory=dict)
    human_benchmark: dict[str, Any] | None = None


def compute_distribution(scores: list[float]) -> LayerDistribution:
    """Compute statistics from a list of scores."""
    arr = np.array(scores)
    return LayerDistribution(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        max=float(arr.max()),
        n=len(scores),
    )


def compute_thresholds(
    distributions: dict[str, LayerDistribution],
) -> dict[str, float]:
    """Derive grade thresholds from training distributions.

    Uses mean - N*std breakpoints:
    - A: mean - 0.5*std
    - B: mean - 1.5*std
    - C: mean - 2.5*std
    - D: mean - 3.5*std
    """
    overall = distributions.get("overall")
    if overall is None:
        return {"A": 0.85, "B": 0.70, "C": 0.55, "D": 0.40}

    return {
        "A": max(0.0, overall.mean - 0.5 * overall.std),
        "B": max(0.0, overall.mean - 1.5 * overall.std),
        "C": max(0.0, overall.mean - 2.5 * overall.std),
        "D": max(0.0, overall.mean - 3.5 * overall.std),
    }


def save_thresholds(thresholds: dict[str, float], path: Path) -> None:
    """Save frozen thresholds to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2))


def load_thresholds(path: Path) -> dict[str, float]:
    """Load frozen thresholds from JSON."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def compute_human_correlation(
    evaluator_scores: list[float],
    human_labels: list[str],
) -> dict[str, Any]:
    """Compute correlation between evaluator scores and human labels.

    Args:
        evaluator_scores: Evaluator scores for each image.
        human_labels: Human labels: 'pass', 'borderline', or 'fail'.

    Returns:
        Dict with spearman_rho, agreement_rate, confusion_matrix.
    """
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    label_to_ordinal = {"fail": 0, "borderline": 1, "pass": 2}
    ordinals = [label_to_ordinal[label] for label in human_labels]

    spearman_result = spearmanr(evaluator_scores, ordinals)
    rho: float = float(spearman_result.statistic)  # type: ignore[reportUnknownMemberType]
    p_value: float = float(spearman_result.pvalue)  # type: ignore[reportUnknownMemberType]

    # Binary confusion (excluding borderline)
    tp = fp = tn = fn = 0
    borderline_near_threshold = 0
    borderline_total = 0

    for score, label in zip(evaluator_scores, human_labels, strict=True):
        if label == "borderline":
            borderline_total += 1
            if abs(score - 0.70) <= 0.05:
                borderline_near_threshold += 1
            continue
        evaluator_pass = score >= 0.70
        human_pass = label == "pass"
        if evaluator_pass and human_pass:
            tp += 1
        elif evaluator_pass and not human_pass:
            fp += 1
        elif not evaluator_pass and not human_pass:
            tn += 1
        else:
            fn += 1

    return {
        "spearman_rho": rho if not np.isnan(rho) else 0.0,
        "p_value": p_value if not np.isnan(p_value) else 1.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "borderline_agreement_rate": (
            borderline_near_threshold / borderline_total
            if borderline_total > 0
            else None
        ),
    }
