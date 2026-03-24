"""Run calibration: evaluate all training images (self-to-self), compute distributions, freeze thresholds.

Usage: uv run python scripts/run_calibration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from product_fidelity_lab.config import get_settings

os.environ.setdefault("FAL_KEY", get_settings().fal_key)

import numpy as np

from product_fidelity_lab.evaluation.calibration import (
    CalibrationReport,
    LayerDistribution,
    compute_distribution,
    compute_thresholds,
    save_thresholds,
)
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import preload_local
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.storage.run_store import RunStore


async def main() -> None:
    settings = get_settings()
    golden_dir = settings.data_dir / "golden"
    specs_dir = golden_dir / "specs"

    specs = load_all_specs(specs_dir)
    training_specs = [s for s in specs if s.category == "training"]
    validation_specs = [s for s in specs if s.category == "validation"]

    print(f"Training specs: {len(training_specs)}")
    print(f"Validation specs: {len(validation_specs)}")
    print()

    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(db_path=settings.db_path, runs_dir=settings.data_dir / "runs")
    await store.initialize()

    # --- Phase 1: Self-to-self on all training images ---
    print("=" * 60)
    print("PHASE 1: Training set self-evaluation (ceiling)")
    print("=" * 60)

    training_scores: dict[str, list[float]] = {
        "overall": [], "afv": [], "depth": [], "brand": [],
    }
    incomplete_training: list[str] = []

    for i, spec in enumerate(training_specs, 1):
        print(f"\n  [{i}/{len(training_specs)}] {spec.shot_id}...", end=" ", flush=True)

        image_path = golden_dir / spec.image_path
        if not image_path.exists():
            print("SKIP (image not found)")
            continue

        # Load depth map
        depth_path = settings.data_dir / "cache" / "golden_depth" / f"{spec.shot_id}.npy"
        if not depth_path.exists():
            print("SKIP (no depth map — run prepare_golden.py)")
            continue
        golden_depth = np.load(str(depth_path))

        # Ensure uploaded
        if spec.image_url is None:
            print("SKIP (no image_url)")
            continue

        # Pre-load from local
        preload_local(spec.image_url, image_path)

        try:
            run = await store.create_run(
                RunType.EVALUATION,
                config={"label": "calibration_training", "spec_id": spec.shot_id},
            )
            report = await run_evaluation(
                spec.image_url, spec, golden_depth,
                fal_client=fal_client,
                gemini_api_key=settings.gemini_api_key,
                gemini_model=settings.gemini_model,
                run_store=store,
                run_id=run.id,
            )
            r = report.model_dump()
            f = r["final"]
            if f["outcome"] == "incomplete" or f["overall"] is None:
                incomplete_training.append(spec.shot_id)
                print(f"INCOMPLETE  {', '.join(f['incomplete_reasons'][:2])}")
                continue
            training_scores["overall"].append(f["overall"])
            training_scores["afv"].append(r["afv"]["score"])
            training_scores["depth"].append(r["depth"]["combined"])
            training_scores["brand"].append(r["brand"]["combined_score"])
            status = "PASS" if f["passed"] else "FAIL"
            print(f"{status}  {f['overall']:.2f} ({f['grade']})  "
                  f"AFV:{r['afv']['score']:.2f} D:{r['depth']['combined']:.3f} B:{r['brand']['combined_score']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # --- Phase 2: Validation set ---
    print()
    print("=" * 60)
    print("PHASE 2: Validation set self-evaluation (held-out)")
    print("=" * 60)

    validation_scores: dict[str, list[float]] = {
        "overall": [], "afv": [], "depth": [], "brand": [],
    }
    incomplete_validation: list[str] = []

    for i, spec in enumerate(validation_specs, 1):
        print(f"\n  [{i}/{len(validation_specs)}] {spec.shot_id}...", end=" ", flush=True)

        image_path = golden_dir / spec.image_path
        depth_path = settings.data_dir / "cache" / "golden_depth" / f"{spec.shot_id}.npy"

        if not image_path.exists() or not depth_path.exists() or spec.image_url is None:
            print("SKIP")
            continue

        golden_depth = np.load(str(depth_path))
        preload_local(spec.image_url, image_path)

        try:
            run = await store.create_run(
                RunType.EVALUATION,
                config={"label": "calibration_validation", "spec_id": spec.shot_id},
            )
            report = await run_evaluation(
                spec.image_url, spec, golden_depth,
                fal_client=fal_client,
                gemini_api_key=settings.gemini_api_key,
                gemini_model=settings.gemini_model,
                run_store=store,
                run_id=run.id,
            )
            r = report.model_dump()
            f = r["final"]
            if f["outcome"] == "incomplete" or f["overall"] is None:
                incomplete_validation.append(spec.shot_id)
                print(f"INCOMPLETE  {', '.join(f['incomplete_reasons'][:2])}")
                continue
            validation_scores["overall"].append(f["overall"])
            validation_scores["afv"].append(r["afv"]["score"])
            validation_scores["depth"].append(r["depth"]["combined"])
            validation_scores["brand"].append(r["brand"]["combined_score"])
            status = "PASS" if f["passed"] else "FAIL"
            print(f"{status}  {f['overall']:.2f} ({f['grade']})  "
                  f"AFV:{r['afv']['score']:.2f} D:{r['depth']['combined']:.3f} B:{r['brand']['combined_score']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # --- Compute distributions and thresholds ---
    print()
    print("=" * 60)
    print("DISTRIBUTIONS & THRESHOLDS")
    print("=" * 60)

    if incomplete_training:
        print("  ERROR: calibration had incomplete training evaluations.")
        print(f"  Incomplete training specs: {', '.join(incomplete_training)}")
        print("  Re-run calibration after the failing layer(s) are healthy.")
        return

    training_dists: dict[str, LayerDistribution] = {}
    for key, scores in training_scores.items():
        if scores:
            dist = compute_distribution(scores)
            training_dists[key] = dist
            print(f"  Training {key:>8}: mean={dist.mean:.3f} std={dist.std:.3f} "
                  f"min={dist.min:.3f} max={dist.max:.3f} n={dist.n}")

    print()
    validation_dists: dict[str, list[float]] = {}
    for key, scores in validation_scores.items():
        if scores:
            validation_dists[key] = scores
            arr = np.array(scores)
            print(f"  Validation {key:>8}: mean={arr.mean():.3f} std={arr.std():.3f} n={len(scores)}")

    thresholds = compute_thresholds(training_dists)
    print()
    print("  Frozen thresholds:")
    for grade, val in thresholds.items():
        print(f"    {grade}: {val:.3f}")

    # Save
    cal_dir = settings.data_dir / "calibration"
    save_thresholds(thresholds, cal_dir / "thresholds.json")

    cal_report = CalibrationReport(
        training_distributions=training_dists,
        frozen_thresholds=thresholds,
        validation_scores=validation_dists,
    )
    (cal_dir / "calibration_report.json").write_text(
        cal_report.model_dump_json(indent=2)
    )

    print(f"\n  Thresholds saved to {cal_dir / 'thresholds.json'}")
    print(f"  Full report saved to {cal_dir / 'calibration_report.json'}")

    # Flag discrepancies
    if training_dists.get("overall") and validation_dists.get("overall"):
        t_mean = training_dists["overall"].mean
        v_mean = float(np.mean(validation_dists["overall"]))
        gap = abs(t_mean - v_mean)
        if gap > 0.1:
            print(f"\n  ⚠ Training/validation gap: {gap:.3f} — review specs for consistency")
        else:
            print(f"\n  Training/validation gap: {gap:.3f} — within expected range")

    if incomplete_validation:
        print(f"\n  Validation incomplete for: {', '.join(incomplete_validation)}")


if __name__ == "__main__":
    asyncio.run(main())
