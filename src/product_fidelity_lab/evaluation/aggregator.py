"""Score aggregation with hard gates and grading."""

from __future__ import annotations

from product_fidelity_lab.models.evaluation import (
    GRADE_THRESHOLDS,
    PASS_THRESHOLD,
    WEIGHTS,
    AFVReport,
    BrandReport,
    DepthScore,
    EvaluationReport,
    FinalScore,
)


def compute_grade(score: float) -> str:
    """Map a score to a letter grade."""
    for grade, threshold in sorted(
        GRADE_THRESHOLDS.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if score >= threshold:
            return grade
    return "F"


def aggregate(
    afv: AFVReport,
    depth: DepthScore,
    brand: BrandReport,
    *,
    grade_thresholds: dict[str, float] | None = None,
    pass_threshold: float | None = None,
) -> FinalScore:
    """Aggregate layer scores, apply hard gates, and produce final score."""
    breakdown = {
        "afv": afv.score,
        "depth": depth.combined,
        "brand": brand.combined_score,
    }

    incomplete_layers: list[str] = []
    incomplete_reasons: list[str] = []

    if afv.error:
        incomplete_layers.append("afv")
        incomplete_reasons.append(afv.error)
    if depth.error:
        incomplete_layers.append("depth")
        incomplete_reasons.append(depth.error)
    if brand.error:
        incomplete_layers.append("brand")
        incomplete_reasons.append(brand.error)

    if incomplete_reasons:
        return FinalScore(
            overall=None,
            grade=None,
            passed=None,
            outcome="incomplete",
            incomplete_reasons=incomplete_reasons,
            incomplete_layers=incomplete_layers,
            breakdown=breakdown,
            confidence=0.0,
        )

    hard_gate_failures: list[str] = []

    # Hard gate 1: critical atomic facts
    hard_gate_failures.extend(afv.critical_failures)

    # Hard gate 2: critical brand text
    hard_gate_failures.extend(brand.critical_failures)

    thresholds = grade_thresholds or GRADE_THRESHOLDS
    effective_pass_threshold = (
        pass_threshold if pass_threshold is not None else thresholds.get("B", PASS_THRESHOLD)
    )

    overall = sum(WEIGHTS[k] * breakdown[k] for k in WEIGHTS)
    grade = compute_grade_from_thresholds(overall, thresholds)

    passed = overall >= effective_pass_threshold and len(hard_gate_failures) == 0

    return FinalScore(
        overall=overall,
        grade=grade,
        passed=passed,
        outcome="graded",
        hard_gate_failures=hard_gate_failures,
        breakdown=breakdown,
    )


def build_report(
    afv: AFVReport,
    depth: DepthScore,
    brand: BrandReport,
    *,
    grade_thresholds: dict[str, float] | None = None,
    pass_threshold: float | None = None,
    run_metadata: dict[str, object] | None = None,
) -> EvaluationReport:
    """Build the full evaluation report from layer results."""
    final = aggregate(
        afv,
        depth,
        brand,
        grade_thresholds=grade_thresholds,
        pass_threshold=pass_threshold,
    )
    return EvaluationReport(
        afv=afv,
        depth=depth,
        brand=brand,
        final=final,
        run_metadata=run_metadata or {},
    )


def compute_grade_from_thresholds(
    score: float,
    thresholds: dict[str, float],
) -> str:
    """Map a score to a letter grade using explicit thresholds."""
    for grade, threshold in sorted(
        thresholds.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if score >= threshold:
            return grade
    return "F"
