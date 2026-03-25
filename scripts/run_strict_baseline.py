"""Strict benchmark baseline: one-shot generation under the strict protocol.

4 runs total — just enough to establish what one-shot can do.

Usage:
  uv run python scripts/run_strict_baseline.py --dry-run
  uv run python scripts/run_strict_baseline.py
"""

from __future__ import annotations

import asyncio
import hashlib
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

from product_fidelity_lab.benchmark_protocol import load_protocol
from product_fidelity_lab.evaluation.calibration import load_thresholds
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import preload_local
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.generation.cold_start import cold_generate
from product_fidelity_lab.generation.prompts import build_prompt_with_strategy
from product_fidelity_lab.models.generation import GenerationRequest
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.search import write_leaderboard_csv
from product_fidelity_lab.storage.run_store import RunStore

PROTOCOL_PATH = Path("data/benchmarks/strict_hero_front_straight.json")

# One ref pack using only strict-allowed shots
REFERENCE_SHOT_IDS = [
    "label_closeup",
    "cap_closeup",
    "glass_material_detail",
    "side_left",
]

PROMPT_STRATEGIES = ["baseline", "label_emphasis"]
SEEDS = [42, 1337]
COST_PER_RUN_EST = 0.12


def _derive_seed(experiment_seed: int, label: str, idx: int) -> int:
    h = hashlib.sha256(f"{experiment_seed}:{label}:{idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


async def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    settings = get_settings()
    protocol = load_protocol(PROTOCOL_PATH)
    protocol.validate_references(REFERENCE_SHOT_IDS)

    print(f"Protocol: {protocol.benchmark_id}")
    print(f"  Refs: {REFERENCE_SHOT_IDS} (verified)")
    print()

    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: thresholds not found.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    all_specs = load_all_specs(settings.data_dir / "golden" / "specs")
    target_spec = next(
        (s for s in all_specs if s.shot_id == protocol.target_shot_id),
        None,
    )
    if target_spec is None or target_spec.image_url is None:
        print("ERROR: target spec missing")
        return

    ref_urls = protocol.select_references(
        all_specs, shot_ids=REFERENCE_SHOT_IDS, max_refs=4,
    )

    experiment_seed = int.from_bytes(
        hashlib.sha256(b"strict-baseline-v1").digest()[:4], "big",
    )

    # 2 prompts × 2 seeds = 4 runs
    grid: list[dict] = []
    for ps in PROMPT_STRATEGIES:
        prompt = build_prompt_with_strategy(
            target_spec, len(ref_urls), strategy=ps,
        )
        for si in range(len(SEEDS)):
            label = f"{ps}__strict_refs__gs3.5"
            seed = _derive_seed(experiment_seed, label, si)
            grid.append({
                "prompt_strategy": ps,
                "seed": seed,
                "prompt_text": prompt,
                "label": label,
                "dir_name": f"{label}_s{seed}",
            })

    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    output_dir = (
        settings.data_dir / "runs" / "experiments"
        / f"{ts}_strict-baseline-v1"
    )

    print("=" * 70)
    print("Product Fidelity Lab — Strict Baseline")
    print("=" * 70)
    print(f"  Runs:   {len(grid)} (${len(grid) * COST_PER_RUN_EST:.2f})")
    print(f"  Refs:   {REFERENCE_SHOT_IDS}")
    print(f"  Output: {output_dir}/")
    print()

    if dry_run:
        print("DRY RUN:")
        for c in grid:
            print(f"  {c['dir_name']}")
        return

    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(
        db_path=settings.db_path, runs_dir=settings.data_dir / "runs",
    )
    await store.initialize()

    data_dir = settings.data_dir
    depth_path = data_dir / "cache" / "golden_depth" / "hero_front_straight.npy"
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        golden_depth = await get_depth_map(target_spec.image_url, fal_client)

    preload_local(
        target_spec.image_url,
        data_dir / "golden" / target_spec.image_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "strict-baseline-v1",
        "timestamp": ts,
        "protocol": protocol.model_dump(),
        "reference_shot_ids": REFERENCE_SHOT_IDS,
        "reference_urls": ref_urls,
        "experiment_seed": experiment_seed,
        "evaluator_snapshot": {
            "grade_thresholds": grade_thresholds,
            "pass_threshold": 0.70,
            "gemini_model": getattr(settings, "gemini_model", "gemini-2.5-flash"),
            "spec_hash": hashlib.sha256(
                target_spec.model_dump_json().encode(),
            ).hexdigest()[:16],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("GENERATION (strict baseline)")
    print("-" * 70)

    results: list[dict] = []
    total_cost = 0.0
    gemini_api_key = getattr(settings, "gemini_api_key", "")
    gemini_model = getattr(settings, "gemini_model", "gemini-2.5-flash")

    for i, config in enumerate(grid, 1):
        print(f"  [{i}/{len(grid)}] {config['dir_name']}...", end=" ", flush=True)

        try:
            gen_request = GenerationRequest(
                prompt=config["prompt_text"],
                reference_urls=ref_urls,
                image_size="square_hd",
                guidance_scale=3.5,
                num_inference_steps=28,
                seed=config["seed"],
            )

            gen_run = await store.create_run(
                RunType.GENERATION, config=gen_request.model_dump(),
            )
            gen_result = await cold_generate(gen_request, fal_client)

            eval_run = await store.create_run(
                RunType.EVALUATION,
                config={
                    "image_url": gen_result.image_url,
                    "spec_id": protocol.target_shot_id,
                    "generation_run_id": gen_run.id,
                    "benchmark": protocol.benchmark_id,
                },
            )

            eval_start = time.monotonic()
            report = await run_evaluation(
                gen_result.image_url,
                target_spec,
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

            rd = report.model_dump()
            final = rd["final"]
            text_score = rd["brand"]["text_score"]["score"]

            run_dir = output_dir / "runs" / config["dir_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            combined = {
                "benchmark": protocol.benchmark_id,
                "config": {
                    "prompt_strategy": config["prompt_strategy"],
                    "reference_shot_ids": REFERENCE_SHOT_IDS,
                    "guidance_scale": 3.5,
                    "seed": gen_result.seed,
                    "prompt_text": config["prompt_text"],
                    "resolved_reference_urls": ref_urls,
                },
                "generation": {
                    "run_id": gen_run.id,
                    "image_url": gen_result.image_url,
                    "seed": gen_result.seed,
                    "model": gen_result.model_id,
                    "duration_ms": gen_result.duration_ms,
                    "cost": gen_result.cost_estimate,
                },
                "evaluation": {
                    "run_id": eval_run.id,
                    "duration_ms": int(eval_duration * 1000),
                    "report": rd,
                },
            }
            (run_dir / "report.json").write_text(json.dumps(combined, indent=2))

            result = {
                "config_label": config["dir_name"],
                "prompt": config["prompt_strategy"],
                "seed": gen_result.seed,
                "image_url": gen_result.image_url,
                "text_score": text_score,
                "brand_score": rd["brand"]["combined_score"],
                "afv_score": rd["afv"]["score"],
                "depth_score": rd["depth"]["combined"],
                "overall": final.get("overall"),
                "grade": final.get("grade"),
                "passed": final.get("passed"),
                "hard_gate_failures": final.get("hard_gate_failures", []),
                "cost": gen_result.cost_estimate,
            }
            results.append(result)
            total_cost += gen_result.cost_estimate

            ov = f"{result['overall']:.3f}" if result["overall"] is not None else "—"
            print(f"text={text_score:.3f} grade={result['grade'] or '—'} overall={ov}")

        except Exception as e:
            print(f"FAILED: {e!s:.80}")
            results.append({"config_label": config["dir_name"], "error": str(e), "text_score": 0.0})

    completed = [r for r in results if "error" not in r]
    completed.sort(key=lambda r: (-(r["overall"] or 0),))

    print()
    print("=" * 70)
    print(f"STRICT BASELINE — {len(completed)} completed, ${total_cost:.2f}")
    print("=" * 70)
    for i, r in enumerate(completed, 1):
        ov = f"{r['overall']:.3f}" if r["overall"] is not None else "—"
        print(f"  {i}. {r['config_label']:<45} grade={r['grade'] or '—'} "
              f"overall={ov} text={r['text_score']:.3f} brand={r.get('brand_score', 0):.3f}")

    summary = {
        "experiment": "strict-baseline-v1",
        "benchmark": protocol.benchmark_id,
        "timestamp": ts,
        "total_runs": len(results),
        "completed": len(completed),
        "total_cost": total_cost,
        "results": completed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    rows = []
    for i, r in enumerate(completed, 1):
        rows.append({
            "rank": i, "prompt": r.get("prompt", ""), "seed": r.get("seed", ""),
            "text_score": f"{r['text_score']:.3f}",
            "brand_score": f"{r.get('brand_score', 0):.3f}",
            "afv_score": f"{r.get('afv_score', 0):.3f}",
            "depth_score": f"{r.get('depth_score', 0):.3f}",
            "grade": r.get("grade") or "", "overall": f"{r['overall']:.3f}" if r["overall"] is not None else "",
            "image_url": r.get("image_url", ""),
            "hard_gates": "; ".join(r.get("hard_gate_failures", [])),
        })
    if rows:
        write_leaderboard_csv(rows, output_dir / "leaderboard.csv")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
