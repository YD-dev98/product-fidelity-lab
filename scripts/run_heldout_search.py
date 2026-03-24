"""Held-out synthesis benchmark: generate hero_front_straight without seeing it.

Stage 1: one-shot generation from non-target references only.
Stage 2 (if needed): edit remediation, still without the target image.

Usage:
  uv run python scripts/run_heldout_search.py --dry-run
  uv run python scripts/run_heldout_search.py
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
from product_fidelity_lab.generation.prompts import (
    build_prompt_with_strategy,
)
from product_fidelity_lab.models.generation import GenerationRequest
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.search import (
    write_leaderboard_csv,
)
from product_fidelity_lab.storage.run_store import RunStore

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

PROTOCOL_PATH = Path("data/benchmarks/held_out_hero_front_straight.json")

PROMPT_STRATEGIES = ["baseline", "label_emphasis"]

# Reference packs — all exclude hero_front_straight
REFERENCE_PACKS = {
    "angles_plus_details": [
        "hero_front_left_45",
        "hero_front_right_45",
        "label_closeup",
        "cap_closeup",
    ],
    "label_heavy_no_target": [
        "label_closeup",
        "hero_front_left_45",
        "hero_front_right_45",
        "glass_material_detail",
    ],
}

GUIDANCE_SCALES = [3.5, 7.0]
SEEDS = [42, 1337]  # shared across all configs
COST_PER_RUN_EST = 0.12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_seed(experiment_seed: int, label: str, seed_idx: int) -> int:
    h = hashlib.sha256(
        f"{experiment_seed}:{label}:{seed_idx}".encode(),
    ).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    settings = get_settings()

    # Load protocol
    protocol = load_protocol(PROTOCOL_PATH)
    print(f"Protocol: {protocol.benchmark_id} ({protocol.mode})")
    print(f"  Target: {protocol.target_shot_id}")
    print(f"  Forbidden: {protocol.forbidden_shot_ids}")
    print(f"  Allowed refs: {protocol.allowed_reference_shot_ids}")
    print()

    # Load specs and thresholds
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
    if target_spec is None:
        print(f"ERROR: target spec {protocol.target_shot_id!r} not found")
        return

    # Build reference packs with protocol enforcement
    resolved_ref_packs: dict[str, list[str]] = {}
    for pack_name, shot_ids in REFERENCE_PACKS.items():
        protocol.validate_references(shot_ids)
        urls = protocol.select_references(
            all_specs, shot_ids=shot_ids, max_refs=4,
        )
        resolved_ref_packs[pack_name] = urls

    # Build grid
    experiment_seed = int.from_bytes(
        hashlib.sha256(b"heldout-hero-v1").digest()[:4], "big",
    )

    grid: list[dict] = []
    for ps in PROMPT_STRATEGIES:
        for rp_name in REFERENCE_PACKS:
            for gs in GUIDANCE_SCALES:
                for si, _ in enumerate(SEEDS):
                    ref_urls = resolved_ref_packs[rp_name]
                    base_label = f"{ps}__{rp_name}__gs{gs:g}"
                    seed = _derive_seed(experiment_seed, base_label, si)
                    prompt = build_prompt_with_strategy(
                        target_spec, len(ref_urls), strategy=ps,
                    )
                    grid.append({
                        "prompt_strategy": ps,
                        "reference_pack": rp_name,
                        "guidance_scale": gs,
                        "seed": seed,
                        "prompt_text": prompt,
                        "reference_urls": ref_urls,
                        "label": base_label,
                        "dir_name": f"{base_label}_s{seed}",
                    })

    total_cost_est = len(grid) * COST_PER_RUN_EST
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    output_dir = (
        settings.data_dir / "runs" / "experiments"
        / f"{ts}_heldout-hero-v1"
    )

    # Pre-flight
    print("=" * 70)
    print("Product Fidelity Lab — Held-Out Synthesis Benchmark")
    print("=" * 70)
    print(f"  Target:    {protocol.target_shot_id}")
    print(f"  Runs:      {len(grid)} (${total_cost_est:.2f})")
    print(f"             {len(PROMPT_STRATEGIES)} prompts x "
          f"{len(REFERENCE_PACKS)} ref packs x "
          f"{len(GUIDANCE_SCALES)} guidance x "
          f"{len(SEEDS)} seeds")
    print(f"  Output:    {output_dir}/")
    print()

    if dry_run:
        print("DRY RUN — configs:")
        for c in grid:
            print(f"  {c['dir_name']}")
            print(f"    refs: {[s for s in REFERENCE_PACKS[c['reference_pack']]]}")
        return

    # Setup
    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(
        db_path=settings.db_path, runs_dir=settings.data_dir / "runs",
    )
    await store.initialize()

    # Golden depth map (for evaluation comparison)
    data_dir = settings.data_dir
    depth_path = (
        data_dir / "cache" / "golden_depth"
        / f"{protocol.target_shot_id}.npy"
    )
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        if target_spec.image_url is None:
            print("ERROR: target spec has no image_url for depth")
            return
        golden_depth = await get_depth_map(target_spec.image_url, fal_client)

    if target_spec.image_url:
        golden_image_path = data_dir / "golden" / target_spec.image_path
        preload_local(target_spec.image_url, golden_image_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "experiment_name": "heldout-hero-v1",
        "timestamp": ts,
        "protocol": protocol.model_dump(),
        "experiment_seed": experiment_seed,
        "prompt_strategies": PROMPT_STRATEGIES,
        "reference_packs": {
            k: {"shot_ids": REFERENCE_PACKS[k], "resolved_urls": v}
            for k, v in resolved_ref_packs.items()
        },
        "guidance_scales": GUIDANCE_SCALES,
        "seeds": SEEDS,
        "evaluator_snapshot": {
            "grade_thresholds": grade_thresholds,
            "pass_threshold": 0.70,
            "gemini_model": getattr(
                settings, "gemini_model", "gemini-2.5-flash",
            ),
            "spec_hash": hashlib.sha256(
                target_spec.model_dump_json().encode(),
            ).hexdigest()[:16],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
    )

    # Run
    print("GENERATION (held-out)")
    print("-" * 70)

    results: list[dict] = []
    total_cost = 0.0

    gemini_api_key = getattr(settings, "gemini_api_key", "")
    gemini_model = getattr(settings, "gemini_model", "gemini-2.5-flash")

    for i, config in enumerate(grid, 1):
        print(
            f"  [{i}/{len(grid)}] {config['dir_name']}...",
            end=" ", flush=True,
        )

        try:
            # Generate
            gen_request = GenerationRequest(
                prompt=config["prompt_text"],
                reference_urls=config["reference_urls"],
                image_size="square_hd",
                guidance_scale=config["guidance_scale"],
                num_inference_steps=28,
                seed=config["seed"],
            )

            gen_run = await store.create_run(
                RunType.GENERATION, config=gen_request.model_dump(),
            )
            gen_start = time.monotonic()
            gen_result = await cold_generate(gen_request, fal_client)
            gen_duration = time.monotonic() - gen_start

            # Evaluate
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
            total_ms = int((gen_duration + eval_duration) * 1000)

            report_dict = report.model_dump()
            final = report_dict["final"]
            text_score = report_dict["brand"]["text_score"]["score"]

            # Save per-run report
            run_dir = output_dir / "runs" / config["dir_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            combined = {
                "benchmark": protocol.benchmark_id,
                "config": {
                    "prompt_strategy": config["prompt_strategy"],
                    "reference_pack": config["reference_pack"],
                    "reference_shot_ids": REFERENCE_PACKS[
                        config["reference_pack"]
                    ],
                    "guidance_scale": config["guidance_scale"],
                    "seed": gen_result.seed,
                    "prompt_text": config["prompt_text"],
                    "resolved_reference_urls": config["reference_urls"],
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
                    "report": report_dict,
                },
            }
            (run_dir / "report.json").write_text(
                json.dumps(combined, indent=2),
            )

            result = {
                "config_label": config["dir_name"],
                "prompt": config["prompt_strategy"],
                "ref_pack": config["reference_pack"],
                "guidance": config["guidance_scale"],
                "seed": gen_result.seed,
                "image_url": gen_result.image_url,
                "text_score": text_score,
                "brand_score": report_dict["brand"]["combined_score"],
                "afv_score": report_dict["afv"]["score"],
                "depth_score": report_dict["depth"]["combined"],
                "overall": final.get("overall"),
                "grade": final.get("grade"),
                "passed": final.get("passed"),
                "outcome": final.get("outcome", "graded"),
                "hard_gate_failures": final.get(
                    "hard_gate_failures", [],
                ),
                "cost": gen_result.cost_estimate,
                "duration_ms": total_ms,
            }
            results.append(result)
            total_cost += gen_result.cost_estimate

            ov = (
                f"{result['overall']:.3f}"
                if result["overall"] is not None else "—"
            )
            print(
                f"text={text_score:.3f} "
                f"grade={result['grade'] or '—'} "
                f"overall={ov}",
            )

        except Exception as e:
            print(f"FAILED: {e!s:.80}")
            results.append({
                "config_label": config["dir_name"],
                "prompt": config["prompt_strategy"],
                "ref_pack": config["reference_pack"],
                "error": str(e),
                "text_score": 0.0,
            })

    # Summary
    completed = [r for r in results if "error" not in r]
    completed.sort(key=lambda r: (
        -r["text_score"],
        -(r["overall"] or 0),
    ))

    print()
    print("=" * 70)
    print(f"RESULTS — {len(completed)} completed, ${total_cost:.2f}")
    print("=" * 70)
    print()

    hdr = (f"  {'#':>3}  {'Config':<50} {'Text':>5} "
           f"{'Grade':>5} {'Overall':>7} {'Brand':>6}")
    print(hdr)
    print(f"  {'—'*3}  {'—'*50} {'—'*5} {'—'*5} {'—'*7} {'—'*6}")

    for i, r in enumerate(completed, 1):
        ov = f"{r['overall']:.3f}" if r["overall"] is not None else "  —  "
        print(
            f"  {i:>3}  {r['config_label']:<50} "
            f"{r['text_score']:>5.3f} "
            f"{r.get('grade') or '—':>5} "
            f"{ov:>7} "
            f"{r.get('brand_score', 0):>6.3f}",
        )

    # Check success criteria
    print()
    best_grade = min(
        (r["grade"] for r in completed if r.get("grade")),
        key=lambda g: {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}.get(g, 9),
        default="F",
    )
    best_text = max(
        (r["text_score"] for r in completed), default=0.0,
    )
    best_brand = max(
        (r.get("brand_score", 0) for r in completed), default=0.0,
    )
    no_hg = any(
        not r.get("hard_gate_failures") and r.get("passed")
        for r in completed
    )

    print("SUCCESS CRITERIA:")
    primary_ok = best_grade in ("A", "B") and no_hg
    print(f"  Primary (B+ no hard gates): "
          f"{'PASS' if primary_ok else 'FAIL'} "
          f"(best={best_grade}, clean pass={'yes' if no_hg else 'no'})")
    secondary_ok = best_text > 0.20 and best_brand > 0.50
    print(f"  Secondary (text>0.20 & brand>0.50): "
          f"{'PASS' if secondary_ok else 'FAIL'} "
          f"(text={best_text:.3f}, brand={best_brand:.3f})")
    stretch_ok = best_grade == "A"
    print(f"  Stretch (any A): "
          f"{'PASS' if stretch_ok else 'FAIL'}")

    # Save artifacts
    summary = {
        "experiment": "heldout-hero-v1",
        "benchmark": protocol.benchmark_id,
        "timestamp": ts,
        "total_runs": len(results),
        "completed": len(completed),
        "total_cost": total_cost,
        "success": {
            "primary": primary_ok,
            "secondary": secondary_ok,
            "stretch": stretch_ok,
        },
        "results": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )

    # Leaderboard CSV
    rows = []
    for i, r in enumerate(completed, 1):
        rows.append({
            "rank": i,
            "prompt": r["prompt"],
            "ref_pack": r["ref_pack"],
            "guidance": r["guidance"],
            "seed": r["seed"],
            "text_score": f"{r['text_score']:.3f}",
            "brand_score": f"{r.get('brand_score', 0):.3f}",
            "afv_score": f"{r.get('afv_score', 0):.3f}",
            "depth_score": f"{r.get('depth_score', 0):.3f}",
            "grade": r.get("grade") or "",
            "overall": (
                f"{r['overall']:.3f}"
                if r["overall"] is not None else ""
            ),
            "image_url": r.get("image_url", ""),
            "hard_gates": "; ".join(
                r.get("hard_gate_failures", []),
            ),
        })
    if rows:
        write_leaderboard_csv(rows, output_dir / "leaderboard.csv")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
