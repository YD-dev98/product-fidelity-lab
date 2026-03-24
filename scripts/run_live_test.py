"""Generate and evaluate images against golden specs.

Supports single runs and batch mode with multiple seeds.
Each run is saved to a unique timestamped folder for evidence preservation.

Usage:
  uv run python scripts/run_live_test.py hero_front_straight
  uv run python scripts/run_live_test.py hero_front_straight --batch 5
  uv run python scripts/run_live_test.py hero_front_straight hero_front_left_45 --batch 3
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
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
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.generation.cold_start import cold_generate
from product_fidelity_lab.generation.prompts import build_generation_prompt, select_reference_urls
from product_fidelity_lab.models.generation import GenerationRequest
from product_fidelity_lab.models.golden_spec import GoldenSpec
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.storage.run_store import RunStore


async def run_one(
    shot_id: str,
    spec: GoldenSpec,
    all_specs: list[GoldenSpec],
    fal_client: FalClient,
    store: RunStore,
    settings: object,
    grade_thresholds: dict[str, float],
    thresholds_path: Path,
    output_root: Path,
    seed: int | None = None,
) -> dict[str, object]:
    """Run one generate→evaluate cycle. Returns the combined report dict."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")

    # --- Generate ---
    ref_urls = select_reference_urls(spec, all_specs, max_refs=4)
    prompt = build_generation_prompt(spec, n_references=len(ref_urls))

    gen_request = GenerationRequest(
        prompt=prompt,
        reference_urls=ref_urls,
        image_size="square_hd",
        guidance_scale=3.5,
        num_inference_steps=28,
        seed=seed,
    )

    gen_run = await store.create_run(RunType.GENERATION, config=gen_request.model_dump())
    gen_start = time.monotonic()

    print(f"  [{shot_id}] Generating (refs={len(ref_urls)}, seed={seed or 'random'})...", end=" ", flush=True)

    gen_result = await cold_generate(gen_request, fal_client)
    gen_duration = time.monotonic() - gen_start
    print(f"seed={gen_result.seed} ({gen_duration:.1f}s)")

    # --- Evaluate ---
    data_dir = Path(getattr(settings, "data_dir", "data"))
    depth_path = data_dir / "cache" / "golden_depth" / f"{shot_id}.npy"
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        if spec.image_url is None:
            raise ValueError(f"Spec {shot_id} has no image_url")
        golden_depth = await get_depth_map(spec.image_url, fal_client)

    golden_image_path = data_dir / "golden" / spec.image_path
    if spec.image_url:
        preload_local(spec.image_url, golden_image_path)

    eval_run = await store.create_run(
        RunType.EVALUATION,
        config={
            "image_url": gen_result.image_url,
            "spec_id": shot_id,
            "generation_run_id": gen_run.id,
        },
    )

    gemini_api_key = getattr(settings, "gemini_api_key", "")
    gemini_model = getattr(settings, "gemini_model", "gemini-2.5-flash")

    eval_start = time.monotonic()
    report = await run_evaluation(
        gen_result.image_url,
        spec,
        golden_depth,
        fal_client=fal_client,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        run_store=store,
        run_id=eval_run.id,
        grade_thresholds=grade_thresholds,
        thresholds_source=str(thresholds_path),
        cache_root=data_dir / "cache" / "results",
    )
    eval_duration = time.monotonic() - eval_start

    r = report.model_dump()
    f = r["final"]

    # Print summary line
    if f["outcome"] == "incomplete":
        print(f"  [{shot_id}] => INCOMPLETE (AFV={r['afv']['score']:.2f} D={r['depth']['combined']:.3f}) ({eval_duration:.1f}s)")
    else:
        status = "PASS" if f["passed"] else "FAIL"
        print(f"  [{shot_id}] => {f['grade']} {f['overall']:.3f} {status} (AFV={r['afv']['score']:.2f} D={r['depth']['combined']:.3f} B={r['brand']['combined_score']:.2f}) ({eval_duration:.1f}s)")

    # Save to unique folder
    run_label = f"{ts}_{shot_id}_s{gen_result.seed}"
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, object] = {
        "shot_id": shot_id,
        "timestamp": ts,
        "generation": {
            "run_id": gen_run.id,
            "image_url": gen_result.image_url,
            "seed": gen_result.seed,
            "model": gen_result.model_id,
            "duration_ms": gen_result.duration_ms,
            "cost": gen_result.cost_estimate,
            "prompt": prompt,
            "n_references": len(ref_urls),
        },
        "evaluation": {
            "run_id": eval_run.id,
            "duration_ms": int(eval_duration * 1000),
            "report": r,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(combined, indent=2))

    return combined


async def main() -> None:
    # Parse args: shot_ids and optional --batch N
    args = sys.argv[1:]
    batch_size = 1
    shot_ids: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--batch":
            batch_size = int(args[i + 1])
            i += 2
        else:
            shot_ids.append(args[i])
            i += 1

    if not shot_ids:
        shot_ids = ["hero_front_straight"]

    settings = get_settings()
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: calibration thresholds not found. Run run_calibration.py first.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    specs_dir = settings.data_dir / "golden" / "specs"
    all_specs = load_all_specs(specs_dir)

    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(db_path=settings.db_path, runs_dir=settings.data_dir / "runs")
    await store.initialize()

    output_root = settings.data_dir / "runs" / "live_tests"
    total_runs = len(shot_ids) * batch_size
    total_cost = 0.0

    print("Product Fidelity Lab — Batch Live Test")
    print("=" * 60)
    print(f"Shots: {', '.join(shot_ids)}")
    print(f"Batch size: {batch_size} per shot ({total_runs} total runs)")
    print(f"Output: {output_root}/")
    print("=" * 60)
    print()

    results: list[dict[str, object]] = []

    for shot_id in shot_ids:
        spec = next((s for s in all_specs if s.shot_id == shot_id), None)
        if spec is None:
            print(f"  [{shot_id}] SKIP — spec not found")
            continue
        if spec.image_url is None:
            print(f"  [{shot_id}] SKIP — no image_url")
            continue

        for run_idx in range(batch_size):
            try:
                combined = await run_one(
                    shot_id, spec, all_specs,
                    fal_client, store, settings,
                    grade_thresholds, thresholds_path,
                    output_root,
                )
                results.append(combined)
                gen = combined["generation"]
                total_cost += gen["cost"]  # type: ignore[union-attr]
            except Exception as e:
                print(f"  [{shot_id}] run {run_idx + 1} FAILED: {e}")

    # Summary table
    print()
    print("=" * 60)
    print(f"SUMMARY — {len(results)} runs, ~${total_cost:.2f} generation cost")
    print("=" * 60)

    for r in results:
        ev = r["evaluation"]
        report = ev["report"]  # type: ignore[union-attr]
        f = report["final"]  # type: ignore[index]
        gen = r["generation"]
        seed = gen["seed"]  # type: ignore[union-attr]
        sid = r["shot_id"]
        if f["outcome"] == "incomplete":
            print(f"  {sid:30s} seed={seed:<12} INCOMPLETE")
        else:
            status = "PASS" if f["passed"] else "FAIL"
            print(f"  {sid:30s} seed={seed:<12} {f['grade']} {f['overall']:.3f} {status}")

    print(f"\nAll reports saved to {output_root}/")


if __name__ == "__main__":
    asyncio.run(main())
