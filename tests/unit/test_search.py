from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from product_fidelity_lab.search import (
    EvaluatorSnapshot,
    Manifest,
    Phase2Spec,
    PhaseSpec,
    RunConfig,
    RunResult,
    SearchExperiment,
    config_stats,
    estimate_cost,
    expand_phase,
    extract_result,
    load_manifest,
    rank_results,
    save_manifest,
)

# -- Fixtures ----------------------------------------------------------------


def _phase_1() -> PhaseSpec:
    return PhaseSpec(
        prompt_strategies=["baseline", "label_emphasis"],
        reference_packs=["default", "hero_only"],
        guidance_scales=[3.5],
        seeds_per_config=2,
    )


def _make_config(
    prompt: str = "baseline",
    refs: str = "default",
    gs: float = 3.5,
    seed: int = 42,
    status: str = "pending",
) -> RunConfig:
    return RunConfig(
        prompt_strategy=prompt,
        reference_pack=refs,
        guidance_scale=gs,
        seed=seed,
        status=status,
    )


def _make_result(
    config: RunConfig | None = None,
    text_score: float = 0.5,
    critical_texts_matched: int = 1,
    critical_text_failures: list[str] | None = None,
    overall: float | None = 0.7,
    grade: str | None = "B",
    error: str | None = None,
) -> RunResult:
    if config is None:
        config = _make_config()
    return RunResult(
        config=config,
        text_score=text_score,
        critical_texts_matched=critical_texts_matched,
        critical_text_failures=critical_text_failures or [],
        missing_critical_texts=critical_text_failures or [],
        overall=overall,
        grade=grade,
        error=error,
    )


# -- Grid expansion ----------------------------------------------------------


class TestExpandPhase:
    def test_correct_count(self) -> None:
        phase = _phase_1()
        configs = expand_phase(phase, rng_seed=12345)
        # 2 prompts × 2 refs × 1 guidance × 2 seeds = 8
        assert len(configs) == 8

    def test_seeds_are_deterministic(self) -> None:
        phase = _phase_1()
        run1 = expand_phase(phase, rng_seed=12345)
        run2 = expand_phase(phase, rng_seed=12345)
        seeds1 = [c.seed for c in run1]
        seeds2 = [c.seed for c in run2]
        assert seeds1 == seeds2

    def test_different_rng_seed_gives_different_seeds(self) -> None:
        phase = _phase_1()
        run1 = expand_phase(phase, rng_seed=111)
        run2 = expand_phase(phase, rng_seed=222)
        seeds1 = [c.seed for c in run1]
        seeds2 = [c.seed for c in run2]
        assert seeds1 != seeds2

    def test_config_filter(self) -> None:
        phase = _phase_1()
        configs = expand_phase(phase, rng_seed=12345, config_filter=["baseline__default__gs3.5"])
        assert all(c.label == "baseline__default__gs3.5" for c in configs)
        assert len(configs) == 2  # 2 seeds

    def test_resolved_inputs_populated(self) -> None:
        phase = PhaseSpec(
            prompt_strategies=["baseline"],
            reference_packs=["default"],
            guidance_scales=[3.5],
            seeds_per_config=1,
        )
        configs = expand_phase(
            phase,
            rng_seed=42,
            resolve_prompt=lambda s, n: f"prompt_{s}_{n}",
            resolve_refs=lambda s: [f"ref_{s}_1", f"ref_{s}_2"],
        )
        assert len(configs) == 1
        c = configs[0]
        assert c.prompt_text == "prompt_baseline_2"
        assert c.reference_urls == ["ref_default_1", "ref_default_2"]


# -- Ranking -----------------------------------------------------------------


class TestRanking:
    def test_fewer_critical_failures_wins(self) -> None:
        r1 = _make_result(text_score=0.8, critical_text_failures=["BRAND"])
        r2 = _make_result(text_score=0.3, critical_text_failures=[])
        ranked = rank_results([r1, r2])
        assert ranked[0] is r2  # no failures beats higher text_score

    def test_more_matched_wins_when_failures_equal(self) -> None:
        r1 = _make_result(critical_texts_matched=2, text_score=0.5)
        r2 = _make_result(critical_texts_matched=1, text_score=0.9)
        ranked = rank_results([r1, r2])
        assert ranked[0] is r1  # more matched beats higher text_score

    def test_higher_text_score_wins_when_matched_equal(self) -> None:
        r1 = _make_result(text_score=0.8, overall=0.5)
        r2 = _make_result(text_score=0.6, overall=0.9)
        ranked = rank_results([r1, r2])
        assert ranked[0] is r1  # text_score beats overall

    def test_higher_overall_breaks_tie(self) -> None:
        r1 = _make_result(text_score=0.5, overall=0.8)
        r2 = _make_result(text_score=0.5, overall=0.6)
        ranked = rank_results([r1, r2])
        assert ranked[0] is r1


# -- Cost estimation ---------------------------------------------------------


class TestCostEstimation:
    def test_conservative_estimate(self) -> None:
        assert estimate_cost(10) == pytest.approx(1.20)
        assert estimate_cost(1) == pytest.approx(0.12)

    def test_zero_runs(self) -> None:
        assert estimate_cost(0) == 0.0


# -- Manifest persistence ----------------------------------------------------


class TestManifest:
    def test_round_trip(self) -> None:
        manifest = Manifest(
            experiment_name="test",
            shot_id="hero_front_straight",
            budget_cap=5.0,
            rng_seed=42,
            phase_1=_phase_1(),
            phase_2_spec=Phase2Spec(),
            evaluator=EvaluatorSnapshot(
                grade_thresholds={"A": 0.85, "B": 0.70},
                pass_threshold=0.70,
                gemini_model="gemini-2.5-flash",
                spec_hash="abc123",
            ),
            resolved_runs=[_make_config(status="complete")],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path)
            assert loaded.experiment_name == "test"
            assert loaded.rng_seed == 42
            assert len(loaded.resolved_runs) == 1
            assert loaded.resolved_runs[0].status == "complete"

    def test_resume_skips_complete(self) -> None:
        complete = _make_config(prompt="baseline", refs="default", seed=100, status="complete")
        terminal = _make_config(
            prompt="baseline", refs="default", seed=200, status="failed_terminal",
        )
        retryable = _make_config(
            prompt="baseline", refs="default", seed=300, status="failed_retryable",
        )
        pending = _make_config(prompt="baseline", refs="default", seed=400, status="pending")
        runs = [complete, terminal, retryable, pending]

        skip = {r.dir_name for r in runs if r.status in ("complete", "failed_terminal")}
        retry = {r.dir_name for r in runs if r.status == "failed_retryable"}
        run = {r.dir_name for r in runs if r.status == "pending"}

        assert complete.dir_name in skip
        assert terminal.dir_name in skip
        assert retryable.dir_name in retry
        assert pending.dir_name in run


# -- Config labels -----------------------------------------------------------


class TestRunConfig:
    def test_label_format(self) -> None:
        c = _make_config(prompt="label_emphasis", refs="hero_only", gs=5.0)
        assert c.label == "label_emphasis__hero_only__gs5"

    def test_dir_name_includes_seed(self) -> None:
        c = _make_config(seed=12345)
        assert c.dir_name == "baseline__default__gs3.5_s12345"

    def test_label_is_filesystem_safe(self) -> None:
        c = _make_config()
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            assert ch not in c.label
            assert ch not in c.dir_name


# -- SearchExperiment validation ---------------------------------------------


class TestSearchExperiment:
    def test_valid_experiment(self) -> None:
        exp = SearchExperiment(
            name="test",
            shot_id="hero_front_straight",
            phase_1=PhaseSpec(
                prompt_strategies=["baseline", "label_emphasis"],
                reference_packs=["default"],
                guidance_scales=[3.5],
            ),
        )
        assert exp.phase_1_grid_size == 2
        assert exp.phase_1_total_runs == 2

    def test_unknown_prompt_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt strategy"):
            SearchExperiment(
                name="test",
                shot_id="x",
                phase_1=PhaseSpec(
                    prompt_strategies=["nonexistent"],
                    reference_packs=["default"],
                    guidance_scales=[3.5],
                ),
            )

    def test_unknown_ref_pack_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reference pack"):
            SearchExperiment(
                name="test",
                shot_id="x",
                phase_1=PhaseSpec(
                    prompt_strategies=["baseline"],
                    reference_packs=["nonexistent"],
                    guidance_scales=[3.5],
                ),
            )

    def test_grid_size_calculation(self) -> None:
        exp = SearchExperiment(
            name="test",
            shot_id="x",
            phase_1=PhaseSpec(
                prompt_strategies=["baseline", "label_emphasis", "detailed_label"],
                reference_packs=["default", "hero_only"],
                guidance_scales=[3.5, 5.0],
                seeds_per_config=3,
            ),
        )
        assert exp.phase_1_grid_size == 3 * 2 * 2  # 12
        assert exp.phase_1_total_runs == 12 * 3  # 36


# -- Result extraction -------------------------------------------------------


class TestExtractResult:
    def test_extracts_text_scores(self) -> None:
        config = _make_config()
        report = {
            "final": {
                "overall": 0.7, "grade": "B", "passed": True,
                "outcome": "graded", "hard_gate_failures": [],
            },
            "afv": {"score": 0.85},
            "depth": {"combined": 0.6},
            "brand": {
                "combined_score": 0.5,
                "text_score": {
                    "score": 0.4,
                    "matches": [
                        {
                            "expected": "BRAND", "matched": True,
                            "score": 1.0, "critical": True,
                            "match_mode": "exact_token",
                        },
                        {
                            "expected": "FINE", "matched": False,
                            "score": 0.0, "critical": False,
                            "match_mode": "fuzzy",
                        },
                    ],
                    "extracted_texts": ["BRAND", "OTHER"],
                    "critical_failures": [],
                },
            },
        }
        result = extract_result(config, report, "https://img.jpg", 0.05, 5000)
        assert result.text_score == 0.4
        assert result.critical_texts_matched == 1
        assert result.matched_texts == ["BRAND"]
        assert result.afv_score == 0.85
        assert result.overall == 0.7

    def test_handles_missing_brand(self) -> None:
        config = _make_config()
        report = {
            "final": {"overall": None, "grade": None, "passed": None, "outcome": "incomplete"},
            "afv": {"score": 0.0},
            "depth": {"combined": 0.0},
            "brand": {
                "combined_score": 0.0,
                "text_score": {"score": 0.0, "matches": [], "extracted_texts": []},
            },
        }
        result = extract_result(config, report, "", 0.05, 1000)
        assert result.outcome == "incomplete"
        assert result.text_score == 0.0


# -- Config stats ------------------------------------------------------------


class TestConfigStats:
    def test_groups_by_config_label(self) -> None:
        c1 = _make_config(prompt="baseline", seed=1)
        c2 = _make_config(prompt="baseline", seed=2)
        c3 = _make_config(prompt="label_emphasis", seed=3)
        r1 = _make_result(c1, text_score=0.5)
        r2 = _make_result(c2, text_score=0.7)
        r3 = _make_result(c3, text_score=0.3)
        stats = config_stats([r1, r2, r3])
        assert len(stats) == 2
        baseline_stat = next(s for s in stats if s["config"] == "baseline__default__gs3.5")
        assert baseline_stat["n_runs"] == 2
        assert baseline_stat["best_text"] == 0.7
