"""End-to-end integration test for the full Product Fidelity Lab pipeline.

Tests the complete generate-evaluate loop using mocked API responses.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from product_fidelity_lab.evaluation.aggregator import build_report
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.models.evaluation import (
    AFVReport,
    BrandReport,
    ColorScore,
    DepthScore,
    EvaluationReport,
    FactVerdict,
    TextMatchScore,
)
from product_fidelity_lab.models.golden_spec import (
    ROI,
    AtomicFact,
    ExpectedText,
    GoldenSpec,
)
from product_fidelity_lab.models.run import RunStatus, RunType
from product_fidelity_lab.storage.run_store import RunStore

TEST_URL = "https://example.com/test.png"


@pytest.fixture
def golden_spec() -> GoldenSpec:
    return GoldenSpec(
        shot_id="test_hero",
        image_path="training/test.png",
        image_url=TEST_URL,
        category="training",
        shot_type="hero",
        atomic_facts=[
            AtomicFact(
                id="F1", category="GEOMETRY",
                fact="The bottle is centered in the frame",
                critical=True, importance="high",
            ),
            AtomicFact(
                id="F2", category="LIGHTING",
                fact="Studio lighting with soft shadows",
                importance="medium",
            ),
            AtomicFact(
                id="F3", category="MATERIAL",
                fact="Glass bottle with reflective surface",
                importance="medium",
            ),
            AtomicFact(
                id="F4", category="COLOR",
                fact="Warm amber liquid visible through glass",
                importance="low",
            ),
        ],
        expected_texts=[
            ExpectedText(text="BRAND", critical=False, match_mode="fuzzy"),
        ],
        brand_colors_hex=["#8B4513", "#FFD700"],
        rois=[ROI(x=0.2, y=0.1, width=0.6, height=0.8, label="product")],
        description="Test hero shot for integration testing",
    )


@pytest.fixture
def golden_depth() -> np.ndarray:
    return np.random.default_rng(42).random((256, 256)).astype(np.float32)


@pytest.fixture
async def run_store(tmp_path: Path) -> RunStore:
    store = RunStore(
        db_path=tmp_path / "test.db",
        runs_dir=tmp_path / "runs",
    )
    await store.initialize()
    return store


# --- Mock helpers ---

IMP_WEIGHTS: dict[str, float] = {"high": 1.5, "medium": 1.0, "low": 0.5}


def _mock_afv(facts: list[AtomicFact]) -> AFVReport:
    verdicts = [
        FactVerdict(
            fact_id=f.id, verdict=True,
            confidence=0.92, reasoning=f"Confirmed: {f.fact}",
        )
        for f in facts
    ]
    tw = sum(IMP_WEIGHTS[f.importance] for f in facts)
    ws = sum(IMP_WEIGHTS[f.importance] for f in facts)
    return AFVReport(
        facts=facts, verdicts=verdicts,
        score=ws / tw if tw else 0.0,
        category_breakdown={
            "GEOMETRY": 1.0, "LIGHTING": 1.0,
            "MATERIAL": 1.0, "COLOR": 1.0,
        },
    )


def _mock_depth() -> DepthScore:
    return DepthScore(
        ssim=0.85, correlation=0.90, mse=0.05, combined=0.88,
    )


def _mock_brand() -> BrandReport:
    return BrandReport(
        text_score=TextMatchScore(
            matches=[], score=0.8, extracted_texts=["BRAND"],
        ),
        color_score=ColorScore(
            brand_colors_hex=["#8B4513", "#FFD700"],
            extracted_colors_hex=["#8A4412", "#FFD800"],
            pairs=[], score=0.85,
        ),
        combined_score=0.83,
    )


@contextmanager
def _patch_layers(
    afv: Any = None,
    depth_map: Any = None,
    depth_score: Any = None,
    brand: Any = None,
    afv_side_effect: Any = None,
):
    """Patch all evaluation layer calls."""
    afv_kw: dict[str, Any] = {"new_callable": AsyncMock}
    if afv_side_effect:
        afv_kw["side_effect"] = afv_side_effect
    else:
        afv_kw["return_value"] = afv or _mock_afv([])

    with (
        patch("product_fidelity_lab.evaluation.engine.verify_facts", **afv_kw),
        patch(
            "product_fidelity_lab.evaluation.engine.get_depth_map",
            new_callable=AsyncMock,
            return_value=depth_map,
        ),
        patch(
            "product_fidelity_lab.evaluation.engine.compare_depth",
            return_value=depth_score or _mock_depth(),
        ),
        patch(
            "product_fidelity_lab.evaluation.engine.evaluate_brand",
            new_callable=AsyncMock,
            return_value=brand or _mock_brand(),
        ),
        patch(
            "product_fidelity_lab.evaluation.image_fetch.fetch_image_bytes",
            new_callable=AsyncMock,
            return_value=b"fake",
        ),
    ):
        yield


async def _run_eval(
    spec: GoldenSpec,
    depth: np.ndarray,
    store: RunStore,
    run_id: str,
    *,
    grade_thresholds: dict[str, float] | None = None,
    pass_threshold: float | None = None,
    thresholds_source: str | None = None,
) -> EvaluationReport:
    return await run_evaluation(
        TEST_URL, spec, depth,
        fal_client=AsyncMock(),
        gemini_api_key="test-key",
        gemini_model="gemini-2.5-flash",
        run_store=store,
        run_id=run_id,
        grade_thresholds=grade_thresholds,
        pass_threshold=pass_threshold,
        thresholds_source=thresholds_source,
    )


# --- Tests ---


class TestFullPipelineMocked:

    async def test_evaluation_produces_valid_report(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        afv = _mock_afv(golden_spec.atomic_facts)
        with _patch_layers(
            afv=afv, depth_map=golden_depth, depth_score=_mock_depth(),
        ):
            run = await run_store.create_run(
                RunType.EVALUATION,
                config={"spec_id": golden_spec.shot_id},
            )
            report = await _run_eval(
                golden_spec, golden_depth, run_store, run.id,
            )

        assert isinstance(report, EvaluationReport)
        assert report.final.overall is not None
        assert 0.0 <= report.final.overall <= 1.0
        assert report.final.grade in ("A", "B", "C", "D", "F")
        assert isinstance(report.final.passed, bool)
        assert report.final.outcome == "graded"
        assert len(report.afv.verdicts) == len(golden_spec.atomic_facts)

        updated = await run_store.get_run(run.id)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETE
        assert updated.score is not None

    async def test_high_score_passes(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        afv = _mock_afv(golden_spec.atomic_facts)
        with _patch_layers(afv=afv, depth_map=golden_depth):
            run = await run_store.create_run(RunType.EVALUATION)
            report = await _run_eval(
                golden_spec, golden_depth, run_store, run.id,
            )

        assert report.final.passed is True
        assert report.final.grade in ("A", "B")
        assert report.final.overall >= 0.70

    async def test_custom_thresholds_affect_grade_and_metadata(
        self,
        golden_spec: GoldenSpec,
        golden_depth: np.ndarray,
        run_store: RunStore,
    ) -> None:
        afv = _mock_afv(golden_spec.atomic_facts)
        custom_thresholds = {"A": 0.95, "B": 0.85, "C": 0.70, "D": 0.55}
        thresholds_source = "/tmp/calibration/thresholds.json"

        with _patch_layers(afv=afv, depth_map=golden_depth):
            run = await run_store.create_run(RunType.EVALUATION)
            report = await _run_eval(
                golden_spec,
                golden_depth,
                run_store,
                run.id,
                grade_thresholds=custom_thresholds,
                thresholds_source=thresholds_source,
            )

        assert report.final.grade == "B"
        assert report.final.passed is True
        assert report.run_metadata["grade_thresholds"] == custom_thresholds
        assert report.run_metadata["pass_threshold"] == custom_thresholds["B"]
        assert report.run_metadata["thresholds_source"] == thresholds_source

    async def test_hard_gate_critical_fact_fails(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        verdicts = [
            FactVerdict(
                fact_id="F1", verdict=False,
                confidence=0.95, reasoning="Not centered",
            ),
            FactVerdict(
                fact_id="F2", verdict=True,
                confidence=0.9, reasoning="OK",
            ),
            FactVerdict(
                fact_id="F3", verdict=True,
                confidence=0.9, reasoning="OK",
            ),
            FactVerdict(
                fact_id="F4", verdict=True,
                confidence=0.9, reasoning="OK",
            ),
        ]
        afv = AFVReport(
            facts=golden_spec.atomic_facts,
            verdicts=verdicts, score=0.625,
            category_breakdown={
                "GEOMETRY": 0.0, "LIGHTING": 1.0,
                "MATERIAL": 1.0, "COLOR": 1.0,
            },
            critical_failures=["Critical fact failed: F1"],
        )

        with _patch_layers(afv=afv, depth_map=golden_depth):
            run = await run_store.create_run(RunType.EVALUATION)
            report = await _run_eval(
                golden_spec, golden_depth, run_store, run.id,
            )

        assert report.final.passed is False
        assert len(report.final.hard_gate_failures) > 0

    async def test_layer_failure_graceful_degradation(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        with _patch_layers(
            depth_map=golden_depth,
            afv_side_effect=RuntimeError("API down"),
        ):
            run = await run_store.create_run(RunType.EVALUATION)
            report = await _run_eval(
                golden_spec, golden_depth, run_store, run.id,
            )

        assert report.afv.score == 0.0
        assert report.afv.error is not None
        assert report.afv.verdicts == []
        assert report.final.outcome == "incomplete"
        assert report.final.overall is None
        assert report.final.grade is None
        assert report.final.passed is None
        assert report.final.incomplete_layers == ["afv"]
        assert report.final.incomplete_reasons == ["AFV layer failed: API down"]

        updated = await run_store.get_run(run.id)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETE
        assert updated.score is None
        assert updated.grade is None
        assert updated.passed is None

    async def test_run_store_tracks_full_lifecycle(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        afv = _mock_afv(golden_spec.atomic_facts)
        with _patch_layers(afv=afv, depth_map=golden_depth):
            run = await run_store.create_run(
                RunType.EVALUATION,
                config={"spec_id": "test_hero"},
            )
            initial = await run_store.get_run(run.id)
            assert initial is not None
            assert initial.status == RunStatus.PENDING

            await _run_eval(
                golden_spec, golden_depth, run_store, run.id,
            )

        final = await run_store.get_run(run.id)
        assert final is not None
        assert final.status == RunStatus.COMPLETE
        assert final.score is not None
        assert final.grade is not None
        assert final.passed is not None
        assert final.duration_ms is not None
        assert final.result is not None

        report = EvaluationReport(**final.result)
        assert report.final.overall == final.score

    async def test_multiple_runs_list(
        self, golden_spec: GoldenSpec,
        golden_depth: np.ndarray, run_store: RunStore,
    ) -> None:
        afv = _mock_afv(golden_spec.atomic_facts)
        with _patch_layers(afv=afv, depth_map=golden_depth):
            for _ in range(3):
                run = await run_store.create_run(RunType.EVALUATION)
                await _run_eval(
                    golden_spec, golden_depth, run_store, run.id,
                )

        runs = await run_store.list_runs(type_filter=RunType.EVALUATION)
        assert len(runs) == 3
        assert all(r.status == RunStatus.COMPLETE for r in runs)


class TestAggregatorIntegration:

    def test_build_report_all_passing(
        self, golden_spec: GoldenSpec,
    ) -> None:
        report = build_report(
            _mock_afv(golden_spec.atomic_facts),
            _mock_depth(), _mock_brand(),
        )
        assert isinstance(report, EvaluationReport)
        assert report.final.overall > 0.70
        assert report.final.passed is True
        assert report.final.hard_gate_failures == []

    def test_serialization_roundtrip(
        self, golden_spec: GoldenSpec,
    ) -> None:
        report = build_report(
            _mock_afv(golden_spec.atomic_facts),
            _mock_depth(), _mock_brand(),
            run_metadata={"test": True},
        )
        data = json.loads(report.model_dump_json())
        restored = EvaluationReport(**data)

        assert restored.final.overall == report.final.overall
        assert restored.final.grade == report.final.grade
        assert len(restored.afv.verdicts) == len(report.afv.verdicts)
        assert restored.run_metadata == {"test": True}


class TestReplayStore:

    def test_loads_from_directory(self, tmp_path: Path) -> None:
        from product_fidelity_lab.storage.replay import ReplayStore

        run_dir = tmp_path / "replay" / "test-run-001"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "id": "test-run-001",
            "type": "evaluation",
            "status": "complete",
            "config": {"spec_id": "hero_front_straight"},
            "score": 0.87,
            "grade": "A",
            "passed": True,
            "result": {
                "afv": {
                    "facts": [], "verdicts": [],
                    "score": 0.9, "category_breakdown": {},
                },
                "depth": {
                    "ssim": 0.85, "correlation": 0.9,
                    "mse": 0.05, "combined": 0.88,
                },
                "brand": {
                    "text_score": {"matches": [], "score": 0.8},
                    "color_score": {
                        "brand_colors_hex": [],
                        "extracted_colors_hex": [],
                        "pairs": [], "score": 0.85,
                    },
                    "combined_score": 0.83,
                },
                "final": {
                    "overall": 0.87, "grade": "A",
                    "passed": True, "hard_gate_failures": [],
                    "breakdown": {
                        "afv": 0.9, "depth": 0.88, "brand": 0.83,
                    },
                },
            },
        }))

        store = ReplayStore(tmp_path / "replay")
        assert store.available is True
        assert len(store.list_runs()) == 1
        assert store.get_run("test-run-001")["score"] == 0.87  # type: ignore[index]

    def test_empty_directory(self, tmp_path: Path) -> None:
        from product_fidelity_lab.storage.replay import ReplayStore

        store = ReplayStore(tmp_path / "nonexistent")
        assert store.available is False
        assert store.list_runs() == []
