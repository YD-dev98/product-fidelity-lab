from __future__ import annotations

from product_fidelity_lab.evaluation.aggregator import (
    aggregate,
    compute_grade,
    compute_grade_from_thresholds,
)
from product_fidelity_lab.models.evaluation import (
    AFVReport,
    BrandReport,
    ColorPair,
    ColorScore,
    DepthScore,
    FactVerdict,
    TextMatch,
    TextMatchScore,
)
from product_fidelity_lab.models.golden_spec import AtomicFact


def _make_afv(score: float = 0.9, critical_failures: list[str] | None = None) -> AFVReport:
    return AFVReport(
        facts=[AtomicFact(id="F1", category="GEOMETRY", fact="test", importance="high")],
        verdicts=[FactVerdict(fact_id="F1", verdict=True, confidence=0.9, reasoning="ok")],
        score=score,
        category_breakdown={"GEOMETRY": score},
        critical_failures=critical_failures or [],
    )


def _make_depth(combined: float = 0.8) -> DepthScore:
    return DepthScore(ssim=combined, correlation=combined, mse=1.0 - combined, combined=combined)


def _make_brand(
    combined: float = 0.85,
    critical_failures: list[str] | None = None,
) -> BrandReport:
    return BrandReport(
        text_score=TextMatchScore(
            matches=[TextMatch(
                expected="test", matched=True, score=1.0,
                match_mode="exact_token", critical=False,
            )],
            score=1.0,
        ),
        color_score=ColorScore(
            brand_colors_hex=["#c8a050"],
            extracted_colors_hex=["#c8a050"],
            pairs=[ColorPair(brand_hex="#c8a050", closest_extracted_hex="#c8a050", delta_e=0.0)],
            score=1.0,
        ),
        combined_score=combined,
        critical_failures=critical_failures or [],
    )


class TestGrading:
    def test_grade_a(self) -> None:
        assert compute_grade(0.90) == "A"

    def test_grade_b(self) -> None:
        assert compute_grade(0.75) == "B"

    def test_grade_c(self) -> None:
        assert compute_grade(0.60) == "C"

    def test_grade_d(self) -> None:
        assert compute_grade(0.45) == "D"

    def test_grade_f(self) -> None:
        assert compute_grade(0.30) == "F"

    def test_grade_uses_custom_thresholds(self) -> None:
        thresholds = {"A": 0.95, "B": 0.80, "C": 0.60, "D": 0.40}
        assert compute_grade_from_thresholds(0.82, thresholds) == "B"


class TestAggregator:
    def test_high_scores_pass(self) -> None:
        result = aggregate(_make_afv(0.9), _make_depth(0.85), _make_brand(0.9))
        assert result.passed is True
        assert result.grade in ("A", "B")
        assert result.outcome == "graded"
        assert len(result.hard_gate_failures) == 0

    def test_critical_fact_failure_triggers_hard_gate(self) -> None:
        afv = _make_afv(0.95, critical_failures=["Critical fact failed: bottle visible"])
        result = aggregate(afv, _make_depth(0.9), _make_brand(0.9))
        assert result.passed is False
        assert len(result.hard_gate_failures) == 1
        assert "Critical fact" in result.hard_gate_failures[0]

    def test_critical_text_failure_triggers_hard_gate(self) -> None:
        brand = _make_brand(0.9, critical_failures=['Brand text missing: "Macallan"'])
        result = aggregate(_make_afv(0.9), _make_depth(0.9), brand)
        assert result.passed is False
        assert any("Brand text" in f for f in result.hard_gate_failures)

    def test_low_score_fails(self) -> None:
        result = aggregate(_make_afv(0.3), _make_depth(0.2), _make_brand(0.3))
        assert result.passed is False
        assert result.grade in ("D", "F")

    def test_weighted_calculation(self) -> None:
        result = aggregate(_make_afv(1.0), _make_depth(1.0), _make_brand(1.0))
        assert abs(result.overall - 1.0) < 0.01

    def test_custom_pass_threshold_is_used(self) -> None:
        result = aggregate(
            _make_afv(0.75),
            _make_depth(0.75),
            _make_brand(0.75),
            grade_thresholds={"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6},
        )
        assert result.grade == "C"
        assert result.passed is False

    def test_breakdown_keys(self) -> None:
        result = aggregate(_make_afv(), _make_depth(), _make_brand())
        assert set(result.breakdown.keys()) == {"afv", "depth", "brand"}

    def test_incomplete_when_any_layer_errors(self) -> None:
        afv = _make_afv()
        afv.error = "AFV layer failed: API down"
        result = aggregate(afv, _make_depth(), _make_brand())
        assert result.outcome == "incomplete"
        assert result.overall is None
        assert result.grade is None
        assert result.passed is None
        assert result.incomplete_layers == ["afv"]
        assert result.incomplete_reasons == ["AFV layer failed: API down"]
