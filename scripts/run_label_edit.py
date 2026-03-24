"""Stage-2 label edit experiment.

Uses fal-ai/flux-2-flex/edit to fix labels on the best generated candidates.
Stage A: non-branded prompts (8 runs).
Stage B: branded prompts, only if Stage A text_score stays at 0 (8 more runs).

Usage:
  uv run python scripts/run_label_edit.py --dry-run
  uv run python scripts/run_label_edit.py
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

from product_fidelity_lab.evaluation.calibration import load_thresholds
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import preload_local
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.generation.edit import edit_image
from product_fidelity_lab.models.generation import EditRequest
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.search import (
    write_leaderboard_csv,
)
from product_fidelity_lab.storage.run_store import RunStore

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

BASE_CANDIDATES = [
    {
        "label": "hero_B705",
        "image_url": "https://v3b.fal.media/files/b/0a93687e/"
                     "FY9bElRvsQtGrsVXUFG3n_G9zqZpAI.jpg",
        "source": "label-fidelity-v1 / label_emphasis__hero_only__gs3.5",
        "seed": 3804093830,
        "eval_snapshot": {
            "overall": 0.705, "grade": "B", "afv": 0.893,
            "depth": 0.805, "brand": 0.401, "text_score": 0.0,
        },
    },
    {
        "label": "hero_B701",
        "image_url": "https://v3b.fal.media/files/b/0a9368ac/"
                     "mZrSjNWvZU4bjH7M8lOyf_KARcdwoq.jpg",
        "source": "label-fidelity-v1 / label_emphasis__hero_only__gs3.5",
        "seed": 114759323,
        "eval_snapshot": {
            "overall": 0.701, "grade": "B", "afv": 0.893,
            "depth": 0.750, "brand": 0.378, "text_score": 0.0,
        },
    },
]

PROMPT_VARIANTS = {
    # --- Stage A: non-branded ---
    "preserve_and_fix": (
        "@image1 is the base image. Preserve the bottle shape, pose, cap, "
        "glass, lighting, and background exactly. Correct only the front "
        "label to match the label in @image2 and @image3. The label text "
        "must be sharp and readable. Do not change bottle geometry, cap, "
        "glass color, liquid, camera angle, background, lighting, or "
        "shadow. Only replace the front label artwork and text."
    ),
    "label_transfer": (
        "Edit the front label on the product in @image1 to exactly match "
        "the label from @image2. Use @image3 as overall product reference. "
        "Keep everything else — bottle, cap, glass, liquid, lighting, "
        "background, shadow — completely unchanged. Only replace the front "
        "label artwork and text."
    ),
    # --- Stage B: branded (only if Stage A text stays at 0) ---
    "branded_minimal": (
        "@image1 is the base image. The front label should read BACARDI "
        "SUPERIOR with a bat logo. Match the label layout from @image2. "
        "Do not change bottle geometry, cap, glass color, liquid, camera "
        "angle, background, lighting, or shadow."
    ),
    "branded_reference": (
        "@image1 is the base image. The front label shows the brand name "
        "and product line exactly as printed on the label in @image2. Copy "
        "all text, logo, and layout from @image2 onto the label area of "
        "@image1. Do not change bottle geometry, cap, glass color, liquid, "
        "camera angle, background, lighting, or shadow."
    ),
}

STAGE_A_PROMPTS = ["preserve_and_fix", "label_transfer"]
STAGE_B_PROMPTS = ["branded_minimal", "branded_reference"]
GUIDANCE_SCALES = [3.5, 7.0]
COST_PER_RUN_EST = 0.12  # edit $0.06 + eval ~$0.06


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_label(
    base_label: str, prompt_name: str, gs: float, seed: int,
) -> str:
    return f"{base_label}__{prompt_name}__gs{gs:g}_s{seed}"


def _dir_name(
    base_label: str, prompt_name: str, gs: float, seed: int,
) -> str:
    return _config_label(base_label, prompt_name, gs, seed)


def _derive_seed(experiment_seed: int, label: str) -> int:
    h = hashlib.sha256(f"{experiment_seed}:{label}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# Run one edit + evaluate
# ---------------------------------------------------------------------------


async def run_one_edit(
    base: dict,
    prompt_name: str,
    prompt_text: str,
    guidance_scale: float,
    seed: int,
    spec: object,
    golden_depth: object,
    label_ref_url: str,
    hero_ref_url: str,
    fal_client: FalClient,
    store: RunStore,
    settings: object,
    grade_thresholds: dict,
    thresholds_path: Path,
    output_dir: Path,
) -> dict:
    """Run one edit + evaluate cycle. Returns result dict."""
    reference_urls = [label_ref_url, hero_ref_url]
    image_urls_ordered = [base["image_url"], *reference_urls]

    request = EditRequest(
        base_image_url=base["image_url"],
        reference_urls=reference_urls,
        prompt=prompt_text,
        image_size="square_hd",
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # Generate (edit)
    gen_run = await store.create_run(
        RunType.GENERATION, config=request.model_dump(),
    )
    gen_start = time.monotonic()
    gen_result = await edit_image(request, fal_client)
    gen_duration = time.monotonic() - gen_start

    # Evaluate
    eval_run = await store.create_run(
        RunType.EVALUATION,
        config={
            "image_url": gen_result.image_url,
            "spec_id": spec.shot_id,  # type: ignore[attr-defined]
            "generation_run_id": gen_run.id,
        },
    )

    gemini_api_key = getattr(settings, "gemini_api_key", "")
    gemini_model = getattr(settings, "gemini_model", "gemini-2.5-flash")
    data_dir = Path(getattr(settings, "data_dir", "data"))

    eval_start = time.monotonic()
    report = await run_evaluation(
        gen_result.image_url,
        spec,  # type: ignore[arg-type]
        golden_depth,  # type: ignore[arg-type]
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
    base_snap = base["eval_snapshot"]

    # Compute deltas
    deltas = {}
    if final.get("overall") is not None:
        deltas["overall_delta"] = final["overall"] - base_snap["overall"]
    deltas["afv_delta"] = report_dict["afv"]["score"] - base_snap["afv"]
    deltas["depth_delta"] = (
        report_dict["depth"]["combined"] - base_snap["depth"]
    )
    deltas["brand_delta"] = (
        report_dict["brand"]["combined_score"] - base_snap["brand"]
    )
    text_score = report_dict["brand"]["text_score"]["score"]
    deltas["text_score_delta"] = text_score - base_snap["text_score"]

    # Save report
    dir_name = _dir_name(
        base["label"], prompt_name, guidance_scale, gen_result.seed,
    )
    run_dir = output_dir / "runs" / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "edit_config": {
            "base_candidate": base["label"],
            "base_image_url": base["image_url"],
            "base_source": base["source"],
            "base_eval_snapshot": base_snap,
            "prompt_variant": prompt_name,
            "prompt_text": prompt_text,
            "image_urls_order": image_urls_ordered,
            "guidance_scale": guidance_scale,
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
            "report": report_dict,
        },
        "deltas": deltas,
    }
    (run_dir / "report.json").write_text(json.dumps(combined, indent=2))

    return {
        "config_label": dir_name,
        "base": base["label"],
        "prompt": prompt_name,
        "guidance": guidance_scale,
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
        "hard_gate_failures": final.get("hard_gate_failures", []),
        "deltas": deltas,
        "cost": gen_result.cost_estimate,
        "duration_ms": total_ms,
        "report_dict": report_dict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args

    settings = get_settings()
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: thresholds not found.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    specs_dir = settings.data_dir / "golden" / "specs"
    all_specs = load_all_specs(specs_dir)
    spec = next(
        (s for s in all_specs if s.shot_id == "hero_front_straight"), None,
    )
    if spec is None or spec.image_url is None:
        print("ERROR: hero_front_straight spec not found or missing URL")
        return

    # Reference images
    label_spec = next(
        (s for s in all_specs if s.shot_id == "label_closeup"), None,
    )
    label_ref_url = label_spec.image_url if label_spec else ""
    hero_ref_url = spec.image_url

    if not label_ref_url:
        print("ERROR: label_closeup spec missing image_url")
        return

    experiment_seed = int.from_bytes(
        hashlib.sha256(b"label-edit-v1").digest()[:4], "big",
    )

    # Build grid
    stage_a_grid = [
        (base, pname, gs)
        for base in BASE_CANDIDATES
        for pname in STAGE_A_PROMPTS
        for gs in GUIDANCE_SCALES
    ]
    stage_b_grid = [
        (base, pname, gs)
        for base in BASE_CANDIDATES
        for pname in STAGE_B_PROMPTS
        for gs in GUIDANCE_SCALES
    ]

    stage_a_cost = len(stage_a_grid) * COST_PER_RUN_EST
    stage_b_cost = len(stage_b_grid) * COST_PER_RUN_EST

    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    output_dir = settings.data_dir / "runs" / "experiments" / f"{ts}_label-edit-v1"

    # Pre-flight
    print("=" * 70)
    print("Product Fidelity Lab — Label Edit Experiment")
    print("=" * 70)
    print(f"  Stage A:  {len(stage_a_grid)} runs (${stage_a_cost:.2f})")
    print(f"            {len(BASE_CANDIDATES)} bases × "
          f"{len(STAGE_A_PROMPTS)} prompts × "
          f"{len(GUIDANCE_SCALES)} guidance")
    print(f"  Stage B:  {len(stage_b_grid)} runs (${stage_b_cost:.2f})")
    print("            triggered only if Stage A text_score stays at 0")
    print(f"  Max cost: ${stage_a_cost + stage_b_cost:.2f}")
    print(f"  Output:   {output_dir}/")
    print(f"  Refs:     label={label_ref_url[:60]}...")
    print(f"            hero={hero_ref_url[:60]}...")
    print()

    if dry_run:
        print("DRY RUN — Stage A configs:")
        for base, pname, gs in stage_a_grid:
            seed = _derive_seed(experiment_seed, f"{base['label']}__{pname}__gs{gs:g}")
            print(f"  {_dir_name(base['label'], pname, gs, seed)}")
        print()
        print("Stage B configs (conditional):")
        for base, pname, gs in stage_b_grid:
            seed = _derive_seed(experiment_seed, f"{base['label']}__{pname}__gs{gs:g}")
            print(f"  {_dir_name(base['label'], pname, gs, seed)}")
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

    # Golden depth map
    data_dir = settings.data_dir
    depth_path = data_dir / "cache" / "golden_depth" / "hero_front_straight.npy"
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        golden_depth = await get_depth_map(spec.image_url, fal_client)

    golden_image_path = data_dir / "golden" / spec.image_path
    preload_local(spec.image_url, golden_image_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest = {
        "experiment_name": "label-edit-v1",
        "timestamp": ts,
        "shot_id": "hero_front_straight",
        "experiment_seed": experiment_seed,
        "base_candidates": BASE_CANDIDATES,
        "reference_urls": {
            "label_closeup": label_ref_url,
            "hero_front_straight": hero_ref_url,
        },
        "evaluator_snapshot": {
            "grade_thresholds": grade_thresholds,
            "pass_threshold": 0.70,
            "gemini_model": getattr(settings, "gemini_model", "gemini-2.5-flash"),
            "spec_hash": hashlib.sha256(
                spec.model_dump_json().encode(),
            ).hexdigest()[:16],
        },
        "stage_a_completed": False,
        "stage_a_success": False,
        "stage_b_triggered": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # -----------------------------------------------------------------------
    # Stage A: non-branded
    # -----------------------------------------------------------------------
    print("STAGE A — non-branded prompts")
    print("-" * 70)

    all_results: list[dict] = []
    total_cost = 0.0

    for i, (base, pname, gs) in enumerate(stage_a_grid, 1):
        seed_label = f"{base['label']}__{pname}__gs{gs:g}"
        seed = _derive_seed(experiment_seed, seed_label)
        print(f"  [{i}/{len(stage_a_grid)}] "
              f"{_dir_name(base['label'], pname, gs, seed)}...",
              end=" ", flush=True)

        try:
            result = await run_one_edit(
                base, pname, PROMPT_VARIANTS[pname], gs, seed,
                spec, golden_depth, label_ref_url, hero_ref_url,
                fal_client, store, settings,
                grade_thresholds, thresholds_path, output_dir,
            )
            all_results.append(result)
            total_cost += result["cost"]

            d = result["deltas"]
            ov = f"{result['overall']:.3f}" if result["overall"] is not None else "—"
            print(
                f"text={result['text_score']:.3f} "
                f"grade={result['grade'] or '—'} overall={ov} "
                f"brand_d={d.get('brand_delta', 0):+.3f} "
                f"afv_d={d.get('afv_delta', 0):+.3f}",
            )
        except Exception as e:
            print(f"FAILED: {e!s:.80}")
            all_results.append({
                "config_label": _dir_name(base["label"], pname, gs, seed),
                "base": base["label"],
                "prompt": pname,
                "error": str(e),
                "text_score": 0.0,
            })

    # Stage gate
    stage_a_best_text = max(
        (r["text_score"] for r in all_results if "error" not in r),
        default=0.0,
    )
    stage_a_success = stage_a_best_text > 0.0

    manifest["stage_a_completed"] = True
    manifest["stage_a_success"] = stage_a_success
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    if stage_a_success:
        print(f"Stage A SUCCESS — best text_score: {stage_a_best_text:.3f}")
        print("Skipping Stage B.")
    else:
        print("Stage A: text_score stayed at 0 across all runs.")

    # -----------------------------------------------------------------------
    # Stage B: branded (conditional)
    # -----------------------------------------------------------------------
    if not stage_a_success:
        manifest["stage_b_triggered"] = True
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
        )

        print()
        print("STAGE B — branded prompts (ablation)")
        print("-" * 70)

        for i, (base, pname, gs) in enumerate(stage_b_grid, 1):
            seed_label = f"{base['label']}__{pname}__gs{gs:g}"
            seed = _derive_seed(experiment_seed, seed_label)
            print(f"  [{i}/{len(stage_b_grid)}] "
                  f"{_dir_name(base['label'], pname, gs, seed)}...",
                  end=" ", flush=True)

            try:
                result = await run_one_edit(
                    base, pname, PROMPT_VARIANTS[pname], gs, seed,
                    spec, golden_depth, label_ref_url, hero_ref_url,
                    fal_client, store, settings,
                    grade_thresholds, thresholds_path, output_dir,
                )
                all_results.append(result)
                total_cost += result["cost"]

                d = result["deltas"]
                ov = (f"{result['overall']:.3f}"
                      if result["overall"] is not None else "—")
                print(
                    f"text={result['text_score']:.3f} "
                    f"grade={result['grade'] or '—'} overall={ov} "
                    f"brand_d={d.get('brand_delta', 0):+.3f} "
                    f"afv_d={d.get('afv_delta', 0):+.3f}",
                )
            except Exception as e:
                print(f"FAILED: {e!s:.80}")
                all_results.append({
                    "config_label": _dir_name(
                        base["label"], pname, gs, seed,
                    ),
                    "base": base["label"],
                    "prompt": pname,
                    "error": str(e),
                    "text_score": 0.0,
                })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    completed = [r for r in all_results if "error" not in r]

    print()
    print("=" * 70)
    print(f"RESULTS — {len(completed)} completed, ${total_cost:.2f} spent")
    print("=" * 70)

    # Ranked table
    print()
    print(f"  {'#':>3}  {'Config':<50} {'Text':>5} {'Grade':>5} "
          f"{'Overall':>7} {'BrandD':>7} {'AFVD':>6} {'DepthD':>7}")
    print(f"  {'—'*3}  {'—'*50} {'—'*5} {'—'*5} "
          f"{'—'*7} {'—'*7} {'—'*6} {'—'*7}")

    # Sort by text fidelity
    completed.sort(key=lambda r: (
        -r["text_score"],
        -r.get("brand_score", 0),
        -(r["overall"] or 0),
    ))

    for i, r in enumerate(completed, 1):
        d = r.get("deltas", {})
        ov = f"{r['overall']:.3f}" if r["overall"] is not None else "  —  "
        bd = f"{d.get('brand_delta', 0):+.3f}"
        ad = f"{d.get('afv_delta', 0):+.3f}"
        dd = f"{d.get('depth_delta', 0):+.3f}"
        print(f"  {i:>3}  {r['config_label']:<50} "
              f"{r['text_score']:>5.3f} {r.get('grade') or '—':>5} "
              f"{ov:>7} {bd:>7} {ad:>6} {dd:>7}")

    # Save artifacts
    summary = {
        "experiment": "label-edit-v1",
        "timestamp": ts,
        "total_runs": len(all_results),
        "completed": len(completed),
        "total_cost": total_cost,
        "stage_a_success": stage_a_success,
        "stage_b_triggered": not stage_a_success,
        "results": completed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )

    # Leaderboard CSV with edit-specific columns
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
            "overall": (f"{r['overall']:.3f}"
                        if r["overall"] is not None else ""),
            "image_url": r.get("image_url", ""),
            "hard_gates": "; ".join(r.get("hard_gate_failures", [])),
        })
    if rows:
        write_leaderboard_csv(rows, output_dir / "leaderboard.csv")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
