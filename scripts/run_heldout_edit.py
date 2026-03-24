"""Held-out edit sweep: repair labels without ever seeing the target image.

Uses best held-out generation candidates as base, with only
label_closeup and hero_front_left_45 as edit references.
hero_front_straight is forbidden.

Usage:
  uv run python scripts/run_heldout_edit.py --dry-run
  uv run python scripts/run_heldout_edit.py
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
from product_fidelity_lab.generation.edit import edit_image
from product_fidelity_lab.models.generation import EditRequest
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.search import write_leaderboard_csv
from product_fidelity_lab.storage.run_store import RunStore

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

PROTOCOL_PATH = Path("data/benchmarks/held_out_hero_front_straight.json")

BASE_CANDIDATES = [
    {
        "label": "heldout_B715",
        "image_url": (
            "https://v3b.fal.media/files/b/0a937e01/"
            "rzWtUwPXLjEnL6qEJ0rXw_osZTenNI.jpg"
        ),
        "source": "heldout-hero-v1 / label_emphasis__angles_plus_details__gs7",
        "eval_snapshot": {
            "overall": 0.715, "grade": "B", "afv": 0.893,
            "depth": 0.798, "brand": 0.376, "text_score": 0.0,
        },
    },
    {
        "label": "heldout_C698",
        "image_url": (
            "https://v3b.fal.media/files/b/0a937e0a/"
            "lzPEOrPTugQIlkc9KUhb6_kwvmvcL9.jpg"
        ),
        "source": "heldout-hero-v1 / label_emphasis__label_heavy_no_target__gs3.5",
        "eval_snapshot": {
            "overall": 0.698, "grade": "C", "afv": 0.893,
            "depth": 0.787, "brand": 0.361, "text_score": 0.0,
        },
    },
]

# Edit references — NO hero_front_straight
EDIT_REFERENCE_SHOT_IDS = ["label_closeup", "hero_front_left_45"]

PROMPT_VARIANTS = {
    "preserve_and_fix": (
        "@image1 is the base image. Preserve the bottle shape, pose, cap, "
        "glass, lighting, and background exactly. Correct only the front "
        "label to match the label in @image2. Use @image3 as an additional "
        "angle reference. The label text must be sharp and readable. "
        "Do not change bottle geometry, cap, glass color, liquid, camera "
        "angle, background, lighting, or shadow. Only replace the front "
        "label artwork and text."
    ),
    "label_transfer": (
        "Edit the front label on the product in @image1 to exactly match "
        "the label from @image2. Use @image3 as an additional angle "
        "reference for the product. Keep everything else — bottle, cap, "
        "glass, liquid, lighting, background, shadow — completely "
        "unchanged. Only replace the front label artwork and text."
    ),
}

GUIDANCE_SCALES = [3.5, 7.0]
COST_PER_RUN_EST = 0.12


def _derive_seed(experiment_seed: int, label: str) -> int:
    h = hashlib.sha256(f"{experiment_seed}:{label}".encode()).digest()
    return int.from_bytes(h[:4], "big")


async def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    settings = get_settings()

    # Load and validate protocol
    protocol = load_protocol(PROTOCOL_PATH)
    protocol.validate_references(EDIT_REFERENCE_SHOT_IDS)
    print(f"Protocol: {protocol.benchmark_id} ({protocol.mode})")
    print(f"  Edit refs: {EDIT_REFERENCE_SHOT_IDS} (verified allowed)")
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

    # Resolve edit reference URLs
    ref_urls: list[str] = []
    for sid in EDIT_REFERENCE_SHOT_IDS:
        spec = next((s for s in all_specs if s.shot_id == sid), None)
        if spec and spec.image_url:
            ref_urls.append(spec.image_url)
        else:
            print(f"ERROR: reference {sid!r} missing image_url")
            return

    experiment_seed = int.from_bytes(
        hashlib.sha256(b"heldout-edit-v1").digest()[:4], "big",
    )

    # Build grid: 2 bases × 2 prompts × 2 guidance = 8
    grid: list[dict] = []
    for base in BASE_CANDIDATES:
        for pname in PROMPT_VARIANTS:
            for gs in GUIDANCE_SCALES:
                seed_label = f"{base['label']}__{pname}__gs{gs:g}"
                seed = _derive_seed(experiment_seed, seed_label)
                grid.append({
                    "base": base,
                    "prompt_name": pname,
                    "guidance_scale": gs,
                    "seed": seed,
                    "dir_name": f"{base['label']}__{pname}__gs{gs:g}_s{seed}",
                })

    total_cost_est = len(grid) * COST_PER_RUN_EST
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    output_dir = (
        settings.data_dir / "runs" / "experiments"
        / f"{ts}_heldout-edit-v1"
    )

    # Pre-flight
    print("=" * 70)
    print("Product Fidelity Lab — Held-Out Edit Sweep")
    print("=" * 70)
    print(f"  Runs:      {len(grid)} (${total_cost_est:.2f})")
    print(f"             {len(BASE_CANDIDATES)} bases × "
          f"{len(PROMPT_VARIANTS)} prompts × "
          f"{len(GUIDANCE_SCALES)} guidance")
    print(f"  Edit refs: {EDIT_REFERENCE_SHOT_IDS}")
    print(f"  Output:    {output_dir}/")
    print()

    if dry_run:
        print("DRY RUN — configs:")
        for c in grid:
            print(f"  {c['dir_name']}")
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

    # Golden depth
    data_dir = settings.data_dir
    depth_path = (
        data_dir / "cache" / "golden_depth"
        / f"{protocol.target_shot_id}.npy"
    )
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        golden_depth = await get_depth_map(
            target_spec.image_url, fal_client,
        )

    if target_spec.image_url:
        preload_local(
            target_spec.image_url,
            data_dir / "golden" / target_spec.image_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "experiment_name": "heldout-edit-v1",
        "timestamp": ts,
        "protocol": protocol.model_dump(),
        "experiment_seed": experiment_seed,
        "base_candidates": BASE_CANDIDATES,
        "edit_reference_shot_ids": EDIT_REFERENCE_SHOT_IDS,
        "edit_reference_urls": ref_urls,
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
    print("EDIT (held-out)")
    print("-" * 70)

    results: list[dict] = []
    total_cost = 0.0
    gemini_api_key = getattr(settings, "gemini_api_key", "")
    gemini_model = getattr(settings, "gemini_model", "gemini-2.5-flash")

    for i, config in enumerate(grid, 1):
        base = config["base"]
        pname = config["prompt_name"]
        gs = config["guidance_scale"]
        seed = config["seed"]

        print(
            f"  [{i}/{len(grid)}] {config['dir_name']}...",
            end=" ", flush=True,
        )

        try:
            request = EditRequest(
                base_image_url=base["image_url"],
                reference_urls=ref_urls,
                prompt=PROMPT_VARIANTS[pname],
                image_size="square_hd",
                guidance_scale=gs,
                seed=seed,
            )

            gen_run = await store.create_run(
                RunType.GENERATION, config=request.model_dump(),
            )
            gen_start = time.monotonic()
            gen_result = await edit_image(request, fal_client)
            gen_duration = time.monotonic() - gen_start

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

            rd = report.model_dump()
            final = rd["final"]
            base_snap = base["eval_snapshot"]
            text_score = rd["brand"]["text_score"]["score"]

            deltas = {
                "text_delta": text_score - base_snap["text_score"],
                "brand_delta": (
                    rd["brand"]["combined_score"] - base_snap["brand"]
                ),
                "afv_delta": rd["afv"]["score"] - base_snap["afv"],
                "depth_delta": (
                    rd["depth"]["combined"] - base_snap["depth"]
                ),
            }
            if final.get("overall") is not None:
                deltas["overall_delta"] = (
                    final["overall"] - base_snap["overall"]
                )

            # Save report
            run_dir = output_dir / "runs" / config["dir_name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            combined = {
                "benchmark": protocol.benchmark_id,
                "edit_config": {
                    "base_candidate": base["label"],
                    "base_image_url": base["image_url"],
                    "base_source": base["source"],
                    "base_eval_snapshot": base_snap,
                    "prompt_variant": pname,
                    "prompt_text": PROMPT_VARIANTS[pname],
                    "edit_reference_shot_ids": EDIT_REFERENCE_SHOT_IDS,
                    "image_urls_order": [
                        base["image_url"], *ref_urls,
                    ],
                    "guidance_scale": gs,
                    "seed": gen_result.seed,
                    "model_id": gen_result.model_id,
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
                "deltas": deltas,
            }
            (run_dir / "report.json").write_text(
                json.dumps(combined, indent=2),
            )

            result = {
                "config_label": config["dir_name"],
                "base": base["label"],
                "prompt": pname,
                "guidance": gs,
                "seed": gen_result.seed,
                "image_url": gen_result.image_url,
                "text_score": text_score,
                "brand_score": rd["brand"]["combined_score"],
                "afv_score": rd["afv"]["score"],
                "depth_score": rd["depth"]["combined"],
                "overall": final.get("overall"),
                "grade": final.get("grade"),
                "passed": final.get("passed"),
                "hard_gate_failures": final.get(
                    "hard_gate_failures", [],
                ),
                "deltas": deltas,
                "cost": gen_result.cost_estimate,
                "duration_ms": total_ms,
            }
            results.append(result)
            total_cost += gen_result.cost_estimate

            d = deltas
            ov = (
                f"{result['overall']:.3f}"
                if result["overall"] is not None else "—"
            )
            print(
                f"text={text_score:.3f} "
                f"grade={result['grade'] or '—'} "
                f"overall={ov} "
                f"brand_d={d.get('brand_delta', 0):+.3f} "
                f"afv_d={d.get('afv_delta', 0):+.3f}",
            )

        except Exception as e:
            print(f"FAILED: {e!s:.80}")
            results.append({
                "config_label": config["dir_name"],
                "base": base["label"],
                "prompt": pname,
                "error": str(e),
                "text_score": 0.0,
            })

    # Summary
    completed = [r for r in results if "error" not in r]
    completed.sort(key=lambda r: (
        -r["text_score"],
        -r.get("brand_score", 0),
        -(r["overall"] or 0),
    ))

    print()
    print("=" * 70)
    print(f"RESULTS — {len(completed)} completed, ${total_cost:.2f}")
    print("=" * 70)
    print()
    print(f"  {'#':>3}  {'Config':<55} {'Text':>5} "
          f"{'Grade':>5} {'Overall':>7} "
          f"{'BrandD':>7} {'AFVD':>6} {'DepthD':>7}")
    print(f"  {'—'*3}  {'—'*55} {'—'*5} {'—'*5} "
          f"{'—'*7} {'—'*7} {'—'*6} {'—'*7}")

    for i, r in enumerate(completed, 1):
        d = r.get("deltas", {})
        ov = f"{r['overall']:.3f}" if r["overall"] is not None else "  —  "
        print(
            f"  {i:>3}  {r['config_label']:<55} "
            f"{r['text_score']:>5.3f} "
            f"{r.get('grade') or '—':>5} "
            f"{ov:>7} "
            f"{d.get('brand_delta', 0):>+7.3f} "
            f"{d.get('afv_delta', 0):>+6.3f} "
            f"{d.get('depth_delta', 0):>+7.3f}",
        )

    # Success criteria
    print()
    best_text = max(
        (r["text_score"] for r in completed), default=0.0,
    )
    best_brand_d = max(
        (r.get("deltas", {}).get("brand_delta", 0) for r in completed),
        default=0.0,
    )
    any_afv_fail = any(
        r.get("deltas", {}).get("afv_delta", 0) < -0.10
        for r in completed
    )
    bacardi_matched = best_text > 0

    print("SUCCESS CRITERIA:")
    print(f"  Primary (BACARDI matches): "
          f"{'PASS' if bacardi_matched else 'FAIL'} "
          f"(best text={best_text:.3f})")
    secondary = (
        best_text > 0
        and best_brand_d > 0.15
        and not any_afv_fail
    )
    print(f"  Secondary (text>0 & brand_d>0.15 & no AFV regression): "
          f"{'PASS' if secondary else 'FAIL'} "
          f"(brand_d={best_brand_d:+.3f}, afv_regress={any_afv_fail})")
    print(f"  Failure (text=0 everywhere): "
          f"{'YES' if best_text == 0 else 'NO'}")

    # Save artifacts
    summary = {
        "experiment": "heldout-edit-v1",
        "benchmark": protocol.benchmark_id,
        "timestamp": ts,
        "total_runs": len(results),
        "completed": len(completed),
        "total_cost": total_cost,
        "success": {
            "primary": bacardi_matched,
            "secondary": secondary,
            "failure": best_text == 0,
        },
        "results": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )

    rows = []
    for i, r in enumerate(completed, 1):
        d = r.get("deltas", {})
        rows.append({
            "rank": i,
            "base": r["base"],
            "prompt": r["prompt"],
            "guidance": r["guidance"],
            "seed": r["seed"],
            "text_score": f"{r['text_score']:.3f}",
            "brand_score": f"{r.get('brand_score', 0):.3f}",
            "brand_delta": f"{d.get('brand_delta', 0):+.3f}",
            "afv_delta": f"{d.get('afv_delta', 0):+.3f}",
            "depth_delta": f"{d.get('depth_delta', 0):+.3f}",
            "overall_delta": f"{d.get('overall_delta', 0):+.3f}",
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
