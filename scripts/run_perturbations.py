"""Run the perturbation suite on a golden image.

Applies controlled degradations, uploads each, runs full evaluation,
and reports which failures the evaluator caught.

Usage: uv run python scripts/run_perturbations.py [shot_id]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from product_fidelity_lab.config import get_settings

os.environ.setdefault("FAL_KEY", get_settings().fal_key)

import numpy as np

from product_fidelity_lab.evaluation.calibration import load_thresholds
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import preload_local
from product_fidelity_lab.evaluation.layer_depth import get_depth_map
from product_fidelity_lab.evaluation.perturbation import (
    DEFAULT_PERTURBATIONS,
    PerturbationType,
    apply_perturbation,
    save_perturbed,
)
from product_fidelity_lab.evaluation.spec_loader import load_spec
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.storage.fal_storage import FalStorage
from product_fidelity_lab.storage.run_store import RunStore


async def main() -> None:
    settings = get_settings()
    shot_id = sys.argv[1] if len(sys.argv) > 1 else "hero_front_straight"
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: calibration thresholds not found. Run run_calibration.py first.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    spec_path = settings.data_dir / "golden" / "specs" / f"{shot_id}.json"
    spec = load_spec(spec_path)
    golden_image_path = settings.data_dir / "golden" / spec.image_path

    print(f"Running perturbation suite on: {shot_id}")
    print(f"Image: {golden_image_path}")
    print(f"Facts: {len(spec.atomic_facts)}, ROIs: {len(spec.rois)}")
    print()

    # Setup
    fal_storage = FalStorage(
        cache_file=settings.data_dir / "cache" / "upload_cache.json",
    )
    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(
        db_path=settings.db_path,
        runs_dir=settings.data_dir / "runs",
    )
    await store.initialize()

    # Load golden depth map
    depth_cache = settings.data_dir / "cache" / "golden_depth" / f"{shot_id}.npy"
    if depth_cache.exists():
        golden_depth = np.load(str(depth_cache))
    else:
        golden_url = spec.image_url
        if golden_url is None:
            print("ERROR: spec has no image_url. Run prepare_golden.py first.")
            return
        golden_depth = await get_depth_map(golden_url, fal_client)

    # Run baseline (golden vs golden) first
    print("=" * 60)
    print("BASELINE: Golden vs Golden (self-evaluation)")
    print("=" * 60)
    baseline_report = await _evaluate(
        golden_image_path, spec, golden_depth, fal_storage, fal_client,
        settings, store, "baseline", grade_thresholds, thresholds_path,
    )
    if baseline_report:
        _print_result("BASELINE", baseline_report)
    print()

    # Run each perturbation
    output_dir = settings.data_dir / "runs" / f"perturbation_{shot_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pdef in DEFAULT_PERTURBATIONS:
        ptype = PerturbationType(pdef["type"])
        params = pdef["params"]
        expected = pdef["expected_failures"]

        print("=" * 60)
        print(f"PERTURBATION: {ptype.value}")
        print(f"Expected failures: {expected}")
        print("=" * 60)

        # Apply perturbation
        perturbed_img = apply_perturbation(
            golden_image_path, ptype, params, rois=spec.rois,
        )
        perturbed_path = output_dir / f"{ptype.value}.png"
        save_perturbed(perturbed_img, perturbed_path)

        # Evaluate
        report = await _evaluate(
            perturbed_path, spec, golden_depth, fal_storage, fal_client,
            settings, store, ptype.value, grade_thresholds, thresholds_path,
        )

        if report:
            actual_failures = _extract_failures(report)
            caught = any(
                exp.lower() in " ".join(actual_failures).lower()
                for exp in expected
            )
            # Also check if score dropped significantly from baseline
            baseline_score = (
                baseline_report["final"]["overall"]
                if baseline_report and baseline_report["final"]["overall"] is not None
                else None
            )
            report_score = report["final"]["overall"]
            score_drop = (
                baseline_score - report_score
                if baseline_score is not None and report_score is not None
                else None
            )
            caught = caught or (score_drop is not None and score_drop > 0.1)

            results.append({
                "perturbation": ptype.value,
                "expected": expected,
                "actual_failures": actual_failures,
                "caught": caught,
                "score": report_score,
                "grade": report["final"]["grade"],
                "score_drop": round(score_drop, 3) if score_drop is not None else None,
            })
            _print_result(ptype.value, report)
            score_drop_display = f"{score_drop:.3f}" if score_drop is not None else "n/a"
            print(f"  Score drop: {score_drop_display}  Caught: {'YES' if caught else 'NO'}")
        else:
            results.append({
                "perturbation": ptype.value,
                "expected": expected,
                "actual_failures": ["evaluation failed"],
                "caught": False,
                "score": None,
                "grade": None,
                "score_drop": None,
            })
        print()

    # Summary
    print()
    print("=" * 60)
    print("PERTURBATION SUITE SUMMARY")
    print("=" * 60)
    caught_count = sum(1 for r in results if r["caught"])
    print(f"Detection rate: {caught_count}/{len(results)}")
    print()
    print(f"{'Perturbation':<25} {'Score':>6} {'Drop':>6} {'Grade':>6} {'Caught':>8}")
    print("-" * 57)
    if baseline_report:
        bs = baseline_report["final"]
        baseline_score = f"{bs['overall']:.2f}" if bs["overall"] is not None else "n/a"
        baseline_grade = bs["grade"] or "n/a"
        print(f"{'baseline':<25} {baseline_score:>6} {'—':>6} {baseline_grade:>6} {'—':>8}")
    for r in results:
        score_display = f"{r['score']:.2f}" if isinstance(r["score"], int | float) else "n/a"
        drop_display = f"{r['score_drop']:+.3f}" if isinstance(r["score_drop"], int | float) else "n/a"
        grade_display = r["grade"] or "n/a"
        print(
            f"{r['perturbation']:<25} {score_display:>6} "
            f"{drop_display:>6} {grade_display:>6} "
            f"{'YES' if r['caught'] else 'NO':>8}"
        )

    # Save report
    report_path = output_dir / "perturbation_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nReport saved to {report_path}")


async def _evaluate(
    image_path: Path,
    spec: object,
    golden_depth: object,
    fal_storage: FalStorage,
    fal_client: FalClient,
    settings: object,
    store: RunStore,
    label: str,
    grade_thresholds: dict[str, float],
    thresholds_path: Path,
) -> dict | None:
    """Upload image and run full evaluation."""
    from product_fidelity_lab.models.run import RunType

    try:
        url = await fal_storage.upload_image(image_path)

        # Pre-load into image cache
        preload_local(url, image_path)

        run = await store.create_run(
            RunType.EVALUATION,
            config={"label": label, "spec_id": spec.shot_id},  # type: ignore[union-attr]
        )

        report = await run_evaluation(
            url,
            spec,  # type: ignore[arg-type]
            golden_depth,
            fal_client=fal_client,
            gemini_api_key=settings.gemini_api_key,  # type: ignore[union-attr]
            gemini_model=settings.gemini_model,  # type: ignore[union-attr]
            run_store=store,
            run_id=run.id,
            grade_thresholds=grade_thresholds,
            thresholds_source=str(thresholds_path),
        )
        return report.model_dump()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def _extract_failures(report: dict) -> list[str]:
    """Extract failure categories from a report."""
    failures = []
    for v in report["afv"]["verdicts"]:
        if not v["verdict"]:
            # Find the fact category
            for f in report["afv"]["facts"]:
                if f["id"] == v["fact_id"]:
                    failures.append(f"AFV:{f['category']}")
                    break
    failures.extend(report["final"]["hard_gate_failures"])
    return failures


def _print_result(label: str, report: dict) -> None:
    f = report["final"]
    if f["outcome"] == "incomplete":
        print(f"  Outcome: INCOMPLETE  Reasons: {f['incomplete_reasons'][:2]}")
    else:
        print(f"  Score: {f['overall']:.2f}  Grade: {f['grade']}  Passed: {f['passed']}")
    print(f"  AFV: {report['afv']['score']:.2f}  "
          f"Depth: {report['depth']['combined']:.3f}  "
          f"Brand: {report['brand']['combined_score']:.2f}")
    if f["hard_gate_failures"]:
        print(f"  Hard gates: {f['hard_gate_failures'][:2]}")
    failed_facts = [v for v in report["afv"]["verdicts"] if not v["verdict"]]
    if failed_facts:
        for v in failed_facts[:3]:
            fact = next(
                (fa for fa in report["afv"]["facts"] if fa["id"] == v["fact_id"]),
                None,
            )
            cat = fact["category"] if fact else "?"
            print(f"    FAIL [{cat}] {v['fact_id']}: {v['reasoning'][:70]}")


if __name__ == "__main__":
    asyncio.run(main())
