"""Tests for the sanity filter stage of the fast ranker."""

from __future__ import annotations

from product_fidelity_lab.critic.fast_ranker import build_filter_summary
from product_fidelity_lab.models.preset import Candidate


def _make_candidate(**kwargs: object) -> Candidate:
    defaults = {
        "candidate_id": "c1",
        "image_url": "https://example.com/img.jpg",
        "seed": 42,
        "model_id": "flux",
        "strategy_used": "reference_only",
        "generation_ms": 1000,
        "cost_estimate": 0.05,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)  # type: ignore[arg-type]


class TestBuildFilterSummary:
    def test_all_passed(self) -> None:
        candidates = [
            _make_candidate(candidate_id="c1", filter_passed=True),
            _make_candidate(candidate_id="c2", filter_passed=True),
        ]
        summary = build_filter_summary(candidates)
        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["failed"] == 0
        assert summary["reasons"] == {}

    def test_mixed_results(self) -> None:
        candidates = [
            _make_candidate(candidate_id="c1", filter_passed=True),
            _make_candidate(
                candidate_id="c2",
                filter_passed=False,
                filter_reasons=["blank_output"],
            ),
            _make_candidate(
                candidate_id="c3",
                filter_passed=False,
                filter_reasons=["blank_output", "color_palette_mismatch"],
            ),
        ]
        summary = build_filter_summary(candidates)
        assert summary["total"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["reasons"]["blank_output"] == 2
        assert summary["reasons"]["color_palette_mismatch"] == 1

    def test_all_failed(self) -> None:
        candidates = [
            _make_candidate(
                candidate_id="c1",
                filter_passed=False,
                filter_reasons=["corrupt_or_unreachable"],
            ),
        ]
        summary = build_filter_summary(candidates)
        assert summary["passed"] == 0
        assert summary["failed"] == 1

    def test_none_filter_passed_counts_as_failed(self) -> None:
        """Candidates with filter_passed=None (not yet filtered) count as failed."""
        candidates = [
            _make_candidate(candidate_id="c1", filter_passed=None),
        ]
        summary = build_filter_summary(candidates)
        assert summary["passed"] == 0
        assert summary["failed"] == 1


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _parse_judge_response

        text = '[{"image": 1, "identity": 0.9, "label": 0.8, "style": 0.7, "composition": 0.6}]'
        results = _parse_judge_response(text, 1)
        assert len(results) == 1
        assert results[0]["identity"] == 0.9
        assert results[0]["composition"] == 0.6

    def test_markdown_fenced(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _parse_judge_response

        text = '```json\n[{"identity": 0.5, "label": 0.5, "style": 0.5, "composition": 0.5}]\n```'
        results = _parse_judge_response(text, 1)
        assert len(results) == 1

    def test_malformed_returns_defaults(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _parse_judge_response

        results = _parse_judge_response("not json at all", 3)
        assert len(results) == 3
        assert all(r["identity"] == 0.5 for r in results)

    def test_clamping(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _parse_judge_response

        text = '[{"identity": 1.5, "label": -0.3, "style": 0.7, "composition": "bad"}]'
        results = _parse_judge_response(text, 1)
        assert results[0]["identity"] == 1.0
        assert results[0]["label"] == 0.0
        assert results[0]["composition"] == 0.5  # fallback for non-numeric


class TestComputeRankScore:
    def test_perfect_scores(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _compute_rank_score

        score = _compute_rank_score({
            "identity": 1.0, "label": 1.0, "style": 1.0, "composition": 1.0,
        })
        assert score == 1.0

    def test_weighted_correctly(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _compute_rank_score

        # identity=1.0 (w=0.4), rest=0.0
        score = _compute_rank_score({
            "identity": 1.0, "label": 0.0, "style": 0.0, "composition": 0.0,
        })
        assert score == 0.4

    def test_zero_scores(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import _compute_rank_score

        score = _compute_rank_score({
            "identity": 0.0, "label": 0.0, "style": 0.0, "composition": 0.0,
        })
        assert score == 0.0


class TestBuildJudgeSummary:
    def test_summary_with_scored_candidates(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import build_judge_summary

        candidates = [
            _make_candidate(candidate_id="c1", rank_score=0.9),
            _make_candidate(candidate_id="c2", rank_score=0.7),
            _make_candidate(candidate_id="c3", rank_score=None),
        ]
        summary = build_judge_summary(candidates, "gemini-2.5-flash")
        assert summary["model"] == "gemini-2.5-flash"
        assert summary["scored"] == 2
        assert summary["top_score"] == 0.9

    def test_summary_no_scores(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import build_judge_summary

        candidates = [_make_candidate(candidate_id="c1", rank_score=None)]
        summary = build_judge_summary(candidates, "gemini-2.5-flash")
        assert summary["scored"] == 0
        assert summary["top_score"] is None


class TestCriticalTextCheck:
    def test_text_found(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import (
            _check_critical_text_present,
        )

        assert _check_critical_text_present(
            ["BACARDI"], ["BACARDI SUPERIOR", "750ml"],
        ) is True

    def test_text_found_case_insensitive(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import (
            _check_critical_text_present,
        )

        assert _check_critical_text_present(
            ["Bacardi"], ["bacardi superior"],
        ) is True

    def test_text_missing(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import (
            _check_critical_text_present,
        )

        assert _check_critical_text_present(
            ["BACARDI"], ["SOMETHING ELSE", "750ml"],
        ) is False

    def test_empty_critical_passes(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import (
            _check_critical_text_present,
        )

        assert _check_critical_text_present([], ["anything"]) is True

    def test_empty_extracted_fails(self) -> None:
        from product_fidelity_lab.critic.fast_ranker import (
            _check_critical_text_present,
        )

        assert _check_critical_text_present(["BACARDI"], []) is False


class TestColorPresenceCheck:
    def test_check_color_presence_with_matching_image(self) -> None:
        """A red image should pass when brand palette includes red."""
        from PIL import Image

        from product_fidelity_lab.critic.fast_ranker import _check_color_presence

        red_img = Image.new("RGB", (100, 100), (200, 30, 30))
        assert _check_color_presence(red_img, ["#cc1e1e"]) is True

    def test_check_color_presence_with_mismatch(self) -> None:
        """A solid blue image should fail against a red-only palette."""
        from PIL import Image

        from product_fidelity_lab.critic.fast_ranker import _check_color_presence

        blue_img = Image.new("RGB", (100, 100), (0, 0, 200))
        assert _check_color_presence(blue_img, ["#cc1e1e"]) is False

    def test_empty_brand_colors_passes(self) -> None:
        from PIL import Image

        from product_fidelity_lab.critic.fast_ranker import _check_color_presence

        img = Image.new("RGB", (100, 100), (128, 128, 128))
        assert _check_color_presence(img, []) is True
