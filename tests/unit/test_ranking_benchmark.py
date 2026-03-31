"""Ranking benchmark: verify ranking logic produces correct orderings.

These tests use synthetic Gemini score data (not live API calls) to verify
that the ranking pipeline selects the human-preferred image.
"""

from __future__ import annotations

from product_fidelity_lab.critic.fast_ranker import (
    _compute_rank_score,
    build_filter_summary,
    build_judge_summary,
)
from product_fidelity_lab.models.preset import Candidate


def _make_candidate(
    cid: str,
    *,
    filter_passed: bool = True,
    filter_reasons: list[str] | None = None,
    identity: float | None = None,
    label: float | None = None,
    style: float | None = None,
    composition: float | None = None,
) -> Candidate:
    c = Candidate(
        candidate_id=cid,
        image_url=f"https://example.com/{cid}.jpg",
        seed=int(cid.replace("c", "")),
        model_id="flux",
        strategy_used="reference_only",
        generation_ms=1000,
        cost_estimate=0.05,
        filter_passed=filter_passed,
        filter_reasons=filter_reasons or [],
    )
    if identity is not None:
        c.identity_score = identity
        c.label_score = label
        c.style_score = style
        c.composition_score = composition
        c.rank_score = _compute_rank_score({
            "identity": identity,
            "label": label or 0.5,
            "style": style or 0.5,
            "composition": composition or 0.5,
        })
    return c


class TestRankingBenchmark:
    """Labeled benchmark cases verifying ranking correctness."""

    def test_case1_clear_winner(self) -> None:
        """High identity+label should beat high style alone."""
        best = _make_candidate("c1", identity=0.95, label=0.90, style=0.60, composition=0.70)
        stylish = _make_candidate("c2", identity=0.50, label=0.40, style=0.95, composition=0.90)
        weak = _make_candidate("c3", identity=0.30, label=0.20, style=0.50, composition=0.50)

        ranked = sorted([weak, stylish, best], key=lambda c: c.rank_score or 0, reverse=True)
        assert ranked[0].candidate_id == "c1"
        assert ranked[-1].candidate_id == "c3"

    def test_case2_label_fidelity_matters(self) -> None:
        """When identity is similar, label score should tip the balance."""
        good_label = _make_candidate("c1", identity=0.80, label=0.90, style=0.70, composition=0.70)
        bad_label = _make_candidate("c2", identity=0.82, label=0.30, style=0.80, composition=0.80)

        ranked = sorted([bad_label, good_label], key=lambda c: c.rank_score or 0, reverse=True)
        assert ranked[0].candidate_id == "c1"

    def test_case3_all_filtered_selects_least_bad(self) -> None:
        """Fallback: least-bad candidate has fewest filter reasons."""
        c1 = _make_candidate(
            "c1", filter_passed=False,
            filter_reasons=["blank_output", "color_palette_mismatch"],
        )
        c2 = _make_candidate(
            "c2", filter_passed=False, filter_reasons=["blank_output"],
        )
        c3 = _make_candidate(
            "c3", filter_passed=False,
            filter_reasons=["corrupt_or_unreachable", "blank_output", "no_subject_detected"],
        )

        failed = [c1, c2, c3]
        least_bad = min(failed, key=lambda c: len(c.filter_reasons))
        assert least_bad.candidate_id == "c2"

    def test_case4_filter_then_rank(self) -> None:
        """Full pipeline: filter removes bad candidates, rank orders survivors."""
        passed1 = _make_candidate(
            "c1", filter_passed=True,
            identity=0.90, label=0.85, style=0.75, composition=0.80,
        )
        passed2 = _make_candidate(
            "c2", filter_passed=True,
            identity=0.70, label=0.60, style=0.80, composition=0.75,
        )
        filtered = _make_candidate(
            "c3", filter_passed=False, filter_reasons=["blank_output"],
        )

        all_candidates = [filtered, passed2, passed1]
        survivors = [c for c in all_candidates if c.filter_passed]
        ranked = sorted(survivors, key=lambda c: c.rank_score or 0, reverse=True)
        selected = ranked[:3]

        # c1 should be top, c3 should be excluded
        assert selected[0].candidate_id == "c1"
        assert all(c.filter_passed for c in selected)

    def test_case5_identical_identity_composition_breaks_tie(self) -> None:
        """When identity and label are equal, composition should break the tie."""
        a = _make_candidate("c1", identity=0.80, label=0.80, style=0.70, composition=0.90)
        b = _make_candidate("c2", identity=0.80, label=0.80, style=0.70, composition=0.50)

        ranked = sorted([b, a], key=lambda c: c.rank_score or 0, reverse=True)
        assert ranked[0].candidate_id == "c1"

    def test_trace_summaries_consistent(self) -> None:
        """Verify filter+judge summaries reflect actual candidate state."""
        candidates = [
            _make_candidate(
                "c1", filter_passed=True,
                identity=0.9, label=0.8, style=0.7, composition=0.7,
            ),
            _make_candidate(
                "c2", filter_passed=True,
                identity=0.6, label=0.5, style=0.6, composition=0.6,
            ),
            _make_candidate(
                "c3", filter_passed=False, filter_reasons=["blank_output"],
            ),
            _make_candidate(
                "c4", filter_passed=False,
                filter_reasons=["corrupt_or_unreachable"],
            ),
        ]

        f_summary = build_filter_summary(candidates)
        assert f_summary["passed"] == 2
        assert f_summary["failed"] == 2
        assert f_summary["reasons"]["blank_output"] == 1

        scored = [c for c in candidates if c.filter_passed]
        j_summary = build_judge_summary(scored, "gemini-2.5-flash")
        assert j_summary["scored"] == 2
        assert j_summary["top_score"] is not None
