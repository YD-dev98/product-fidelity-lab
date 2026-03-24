"""Staged search runner for generation experiments.

Phase 1 sweeps prompt strategies × reference packs cheaply.
Phase 2 takes the top configs and varies guidance scale + seeds.

Usage:
  uv run python scripts/run_search.py --dry-run
  uv run python scripts/run_search.py
  uv run python scripts/run_search.py --phase1-only
  uv run python scripts/run_search.py --resume
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
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
from product_fidelity_lab.generation.prompts import (
    build_prompt_with_strategy,
    select_references_with_strategy,
)
from product_fidelity_lab.models.generation import GenerationRequest
from product_fidelity_lab.models.run import RunType
from product_fidelity_lab.search import (
    EvaluatorSnapshot,
    Manifest,
    Phase2Spec,
    PhaseSpec,
    RunConfig,
    RunResult,
    SearchExperiment,
    build_leaderboard_rows,
    build_summary_md,
    config_stats,
    estimate_cost,
    expand_phase,
    extract_result,
    generate_rng_seed,
    load_manifest,
    rank_results,
    save_manifest,
    spec_hash,
    write_leaderboard_csv,
)
from product_fidelity_lab.storage.run_store import RunStore

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

EXPERIMENT = SearchExperiment(
    name="label-fidelity-v1",
    shot_id="hero_front_straight",
    phase_1=PhaseSpec(
        prompt_strategies=["baseline", "label_emphasis", "detailed_label", "reference_focused"],
        reference_packs=["default", "label_heavy", "hero_only", "front_angles"],
        guidance_scales=[3.5],
        seeds_per_config=1,
    ),
    phase_2=Phase2Spec(
        top_n=3,
        guidance_scales=[3.5, 5.0, 7.0],
        seeds_per_config=2,
    ),
    budget_cap=5.00,
)


# ---------------------------------------------------------------------------
# Run one config through generate → evaluate
# ---------------------------------------------------------------------------


async def run_one_config(
    config: RunConfig,
    spec: object,
    golden_depth: object,
    fal_client: FalClient,
    store: RunStore,
    settings: object,
    grade_thresholds: dict[str, float],
    thresholds_path: Path,
    output_dir: Path,
) -> RunResult:
    """Generate and evaluate one config. Returns a RunResult."""
    run_dir = output_dir / "runs" / config.dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gen_request = GenerationRequest(
        prompt=config.prompt_text,
        reference_urls=config.reference_urls,
        image_size=config.image_size,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        seed=config.seed,
    )

    gen_run = await store.create_run(RunType.GENERATION, config=gen_request.model_dump())
    gen_start = time.monotonic()

    gen_result = await cold_generate(gen_request, fal_client)
    gen_duration = time.monotonic() - gen_start

    # Evaluate
    eval_run = await store.create_run(
        RunType.EVALUATION,
        config={
            "image_url": gen_result.image_url,
            "spec_id": spec.shot_id,  # type: ignore[union-attr]
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
    total_duration_ms = int((gen_duration + eval_duration) * 1000)

    report_dict = report.model_dump()

    # Save report with search config metadata
    combined = {
        "search_config": {
            "prompt_strategy": config.prompt_strategy,
            "reference_pack": config.reference_pack,
            "guidance_scale": config.guidance_scale,
            "seed": config.seed,
            "config_label": config.label,
        },
        "generation": {
            "run_id": gen_run.id,
            "image_url": gen_result.image_url,
            "seed": gen_result.seed,
            "model": gen_result.model_id,
            "duration_ms": gen_result.duration_ms,
            "cost": gen_result.cost_estimate,
            "prompt": config.prompt_text,
            "n_references": len(config.reference_urls),
        },
        "evaluation": {
            "run_id": eval_run.id,
            "duration_ms": int(eval_duration * 1000),
            "report": report_dict,
        },
    }
    (run_dir / "report.json").write_text(json.dumps(combined, indent=2))

    return extract_result(
        config=config,
        report=report_dict,
        image_url=gen_result.image_url,
        cost=gen_result.cost_estimate,
        duration_ms=total_duration_ms,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_table_header() -> None:
    print(f"  {'#':>3}  {'Config':<45} {'Seed':<12} {'Text':>5} {'Grade':>5} {'Overall':>7} {'Missing Critical'}")
    print(f"  {'—'*3}  {'—'*45} {'—'*12} {'—'*5} {'—'*5} {'—'*7} {'—'*30}")


def _print_result_row(i: int, r: RunResult) -> None:
    missing = ", ".join(r.missing_critical_texts) if r.missing_critical_texts else "—"
    overall = f"{r.overall:.3f}" if r.overall is not None else "  —  "
    print(f"  {i:>3}  {r.config.label:<45} {r.config.seed:<12} {r.text_score:>5.3f} {r.grade or '—':>5} {overall:>7} {missing}")


async def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args
    resume = "--resume" in args
    phase1_only = "--phase1-only" in args

    experiment = EXPERIMENT
    settings = get_settings()

    # Load thresholds
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        print("ERROR: calibration thresholds not found. Run run_calibration.py first.")
        return
    grade_thresholds = load_thresholds(thresholds_path)

    # Load specs
    specs_dir = settings.data_dir / "golden" / "specs"
    all_specs = load_all_specs(specs_dir)
    spec = next((s for s in all_specs if s.shot_id == experiment.shot_id), None)
    if spec is None:
        print(f"ERROR: spec {experiment.shot_id!r} not found")
        return
    if spec.image_url is None:
        print(f"ERROR: spec {experiment.shot_id!r} has no image_url")
        return

    output_dir = settings.data_dir / "runs" / "experiments" / experiment.name
    manifest_path = output_dir / "manifest.json"

    # Resolve helpers
    def resolve_prompt(strategy: str, n_refs: int) -> str:
        return build_prompt_with_strategy(spec, n_refs, strategy=strategy)

    def resolve_refs(strategy: str) -> list[str]:
        return select_references_with_strategy(spec, all_specs, strategy=strategy)

    # Build or load manifest
    rng_seed = experiment.rng_seed if experiment.rng_seed is not None else generate_rng_seed()

    if resume and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        print(f"Resumed experiment: {manifest.experiment_name}")
        rng_seed = manifest.rng_seed
    else:
        evaluator = EvaluatorSnapshot(
            grade_thresholds=grade_thresholds,
            pass_threshold=0.70,
            gemini_model=getattr(settings, "gemini_model", "gemini-2.5-flash"),
            spec_hash=spec_hash(spec.model_dump_json()),
        )
        manifest = Manifest(
            experiment_name=experiment.name,
            shot_id=experiment.shot_id,
            budget_cap=experiment.budget_cap,
            rng_seed=rng_seed,
            phase_1=experiment.phase_1,
            phase_2_spec=experiment.phase_2,
            evaluator=evaluator,
        )

    # Expand phase 1
    phase_1_configs = expand_phase(
        experiment.phase_1,
        rng_seed,
        resolve_prompt=resolve_prompt,
        resolve_refs=resolve_refs,
    )

    # Pre-flight
    p1_cost = estimate_cost(len(phase_1_configs))
    p2_max_cost = estimate_cost(experiment.phase_2_max_runs)
    total_est = p1_cost + p2_max_cost

    print("=" * 70)
    print(f"Product Fidelity Lab — Search Experiment: {experiment.name}")
    print("=" * 70)
    print(f"  Shot:        {experiment.shot_id}")
    print(f"  Phase 1:     {len(phase_1_configs)} runs (${p1_cost:.2f})")
    print(f"               {len(experiment.phase_1.prompt_strategies)} prompts × "
          f"{len(experiment.phase_1.reference_packs)} refs × "
          f"{len(experiment.phase_1.guidance_scales)} guidance × "
          f"{experiment.phase_1.seeds_per_config} seeds")
    print(f"  Phase 2:     up to {experiment.phase_2_max_runs} runs (${p2_max_cost:.2f})")
    print(f"               top {experiment.phase_2.top_n} × "
          f"{len(experiment.phase_2.guidance_scales)} guidance × "
          f"{experiment.phase_2.seeds_per_config} seeds")
    print(f"  Total est:   ${total_est:.2f} (budget cap: ${experiment.budget_cap:.2f})")
    print(f"  RNG seed:    {rng_seed}")
    print()

    if total_est > experiment.budget_cap:
        print(f"  WARNING: estimated cost ${total_est:.2f} exceeds budget cap ${experiment.budget_cap:.2f}")
        print("           runs will stop when budget is reached")
        print()

    if dry_run:
        print("DRY RUN — no API calls. Configs:")
        for c in phase_1_configs:
            print(f"  {c.dir_name}")
        return

    # Setup
    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )
    store = RunStore(db_path=settings.db_path, runs_dir=settings.data_dir / "runs")
    await store.initialize()

    # Pre-compute golden depth map
    data_dir = settings.data_dir
    depth_path = data_dir / "cache" / "golden_depth" / f"{experiment.shot_id}.npy"
    if depth_path.exists():
        golden_depth = np.load(str(depth_path))
    else:
        from product_fidelity_lab.evaluation.layer_depth import get_depth_map
        golden_depth = await get_depth_map(spec.image_url, fal_client)

    golden_image_path = data_dir / "golden" / spec.image_path
    if spec.image_url:
        preload_local(spec.image_url, golden_image_path)

    # Save initial manifest
    completed_labels = {r.dir_name for r in manifest.resolved_runs if r.status == "complete"}
    terminal_labels = {r.dir_name for r in manifest.resolved_runs if r.status == "failed_terminal"}
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    print("PHASE 1")
    print("-" * 70)

    phase_1_results: list[RunResult] = []
    total_cost = manifest.total_cost

    for i, config in enumerate(phase_1_configs, 1):
        # Skip completed or terminal
        if config.dir_name in completed_labels or config.dir_name in terminal_labels:
            print(f"  [{i}/{len(phase_1_configs)}] {config.dir_name} — skipped (already done)")
            # Load existing result if available
            report_path = output_dir / "runs" / config.dir_name / "report.json"
            if report_path.exists():
                rdata = json.loads(report_path.read_text())
                r = extract_result(
                    config=config,
                    report=rdata["evaluation"]["report"],
                    image_url=rdata["generation"]["image_url"],
                    cost=rdata["generation"]["cost"],
                    duration_ms=0,
                )
                phase_1_results.append(r)
            continue

        # Budget check
        if total_cost + COST_PER_RUN > experiment.budget_cap:
            print(f"  [{i}/{len(phase_1_configs)}] BUDGET REACHED (${total_cost:.2f}/${experiment.budget_cap:.2f})")
            break

        print(f"  [{i}/{len(phase_1_configs)}] {config.dir_name}...", end=" ", flush=True)

        try:
            result = await run_one_config(
                config, spec, golden_depth,
                fal_client, store, settings, grade_thresholds, thresholds_path, output_dir,
            )
            config.status = "complete"
            phase_1_results.append(result)
            total_cost += result.generation_cost

            # Print result
            missing = ", ".join(result.missing_critical_texts) if result.missing_critical_texts else "—"
            ov = f"{result.overall:.3f}" if result.overall is not None else "—"
            print(f"text={result.text_score:.3f} grade={result.grade or '—'} "
                  f"overall={ov} missing=[{missing}]")

        except Exception as e:
            err_str = str(e)
            is_terminal = "safety" in err_str.lower() or "refused" in err_str.lower() or "content" in err_str.lower()
            config.status = "failed_terminal" if is_terminal else "failed_retryable"
            result = RunResult(config=config, error=err_str)
            phase_1_results.append(result)
            status_label = "TERMINAL" if is_terminal else "RETRYABLE"
            print(f"FAILED ({status_label}): {err_str[:80]}")

        # Update manifest
        manifest.resolved_runs = [r for r in manifest.resolved_runs if r.dir_name != config.dir_name]
        manifest.resolved_runs.append(config)
        manifest.total_cost = total_cost
        save_manifest(manifest, manifest_path)

    # Phase 1 ranking
    completed_p1 = [r for r in phase_1_results if r.error is None]
    ranked_p1 = rank_results(completed_p1)

    print()
    print("PHASE 1 RESULTS")
    print("-" * 70)
    _print_table_header()
    for i, r in enumerate(ranked_p1, 1):
        _print_result_row(i, r)

    # Determine phase 2 winners
    top_n = experiment.phase_2.top_n
    seen_labels: set[str] = set()
    winner_labels: list[str] = []
    for r in ranked_p1:
        if r.config.label not in seen_labels:
            seen_labels.add(r.config.label)
            winner_labels.append(r.config.label)
            if len(winner_labels) >= top_n:
                break

    manifest.phase_2_winners = winner_labels
    save_manifest(manifest, manifest_path)

    print()
    print(f"Phase 2 winners: {', '.join(winner_labels)}")

    if phase1_only:
        print()
        print("--phase1-only: stopping here.")
        # Save outputs
        _save_outputs(output_dir, experiment, phase_1_results, total_cost, "phase1")
        return

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------
    print()
    print("PHASE 2")
    print("-" * 70)

    # Build phase 2 spec from winners
    winner_prompts = list({w.split("__")[0] for w in winner_labels})
    winner_refs = list({w.split("__")[1] for w in winner_labels})
    phase_2_phase = PhaseSpec(
        prompt_strategies=winner_prompts,
        reference_packs=winner_refs,
        guidance_scales=experiment.phase_2.guidance_scales,
        seeds_per_config=experiment.phase_2.seeds_per_config,
    )

    phase_2_configs = expand_phase(
        phase_2_phase,
        rng_seed,
        resolve_prompt=resolve_prompt,
        resolve_refs=resolve_refs,
        config_filter=winner_labels,
    )

    phase_2_results: list[RunResult] = []

    # Rebuild completed/terminal sets to include phase 1 results
    completed_labels = {r.dir_name for r in manifest.resolved_runs if r.status == "complete"}
    terminal_labels = {r.dir_name for r in manifest.resolved_runs if r.status == "failed_terminal"}

    for i, config in enumerate(phase_2_configs, 1):
        if config.dir_name in completed_labels or config.dir_name in terminal_labels:
            print(f"  [{i}/{len(phase_2_configs)}] {config.dir_name} — skipped")
            report_path = output_dir / "runs" / config.dir_name / "report.json"
            if report_path.exists():
                rdata = json.loads(report_path.read_text())
                r = extract_result(
                    config=config,
                    report=rdata["evaluation"]["report"],
                    image_url=rdata["generation"]["image_url"],
                    cost=rdata["generation"]["cost"],
                    duration_ms=0,
                )
                phase_2_results.append(r)
            continue

        if total_cost + COST_PER_RUN > experiment.budget_cap:
            print(f"  [{i}/{len(phase_2_configs)}] BUDGET REACHED (${total_cost:.2f}/${experiment.budget_cap:.2f})")
            break

        print(f"  [{i}/{len(phase_2_configs)}] {config.dir_name}...", end=" ", flush=True)

        try:
            result = await run_one_config(
                config, spec, golden_depth,
                fal_client, store, settings, grade_thresholds, thresholds_path, output_dir,
            )
            config.status = "complete"
            phase_2_results.append(result)
            total_cost += result.generation_cost

            missing = ", ".join(result.missing_critical_texts) if result.missing_critical_texts else "—"
            ov = f"{result.overall:.3f}" if result.overall is not None else "—"
            print(f"text={result.text_score:.3f} grade={result.grade or '—'} "
                  f"overall={ov} missing=[{missing}]")

        except Exception as e:
            err_str = str(e)
            is_terminal = "safety" in err_str.lower() or "refused" in err_str.lower() or "content" in err_str.lower()
            config.status = "failed_terminal" if is_terminal else "failed_retryable"
            result = RunResult(config=config, error=err_str)
            phase_2_results.append(result)
            status_label = "TERMINAL" if is_terminal else "RETRYABLE"
            print(f"FAILED ({status_label}): {err_str[:80]}")

        manifest.resolved_runs = [r for r in manifest.resolved_runs if r.dir_name != config.dir_name]
        manifest.resolved_runs.append(config)
        manifest.total_cost = total_cost
        save_manifest(manifest, manifest_path)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    all_results = phase_1_results + phase_2_results
    _save_outputs(output_dir, experiment, all_results, total_cost, "all")

    all_completed = [r for r in all_results if r.error is None]
    ranked_all = rank_results(all_completed)

    print()
    print("=" * 70)
    print(f"FINAL RESULTS — {len(all_completed)} completed, ${total_cost:.2f} spent")
    print("=" * 70)

    # Config-level stats
    stats = config_stats(all_completed)
    print()
    print("CONFIG SUMMARY (sorted by text fidelity):")
    print(f"  {'Config':<45} {'Runs':>4} {'BestText':>8} {'BestBrand':>9} {'BestGrade':>9} {'CritFails':>9}")
    print(f"  {'—'*45} {'—'*4} {'—'*8} {'—'*9} {'—'*9} {'—'*9}")
    for s in stats:
        print(f"  {s['config']:<45} {s['n_runs']:>4} {s['best_text']:>8.3f} "
              f"{s['best_brand']:>9.3f} {s['best_grade'] or '—':>9} {s['total_critical_failures']:>9}")

    print()
    print("TOP 10 RUNS:")
    _print_table_header()
    for i, r in enumerate(ranked_all[:10], 1):
        _print_result_row(i, r)

    # Failure tracking
    failed = [r for r in all_results if r.error is not None]
    if failed:
        print()
        print(f"FAILURES ({len(failed)}):")
        for r in failed:
            print(f"  {r.config.dir_name}: [{r.config.status}] {r.error[:80]}")

    print(f"\nResults saved to {output_dir}/")


COST_PER_RUN = 0.12


def _save_outputs(
    output_dir: Path,
    experiment: SearchExperiment,
    results: list[RunResult],
    total_cost: float,
    phase_label: str,
) -> None:
    completed = [r for r in results if r.error is None]

    # Summary JSON
    summary = {
        "experiment": experiment.name,
        "shot_id": experiment.shot_id,
        "total_runs": len(results),
        "completed_runs": len(completed),
        "failed_runs": len(results) - len(completed),
        "total_cost": total_cost,
        "phase": phase_label,
        "ranked_results": [r.model_dump() for r in rank_results(completed)],
        "config_stats": config_stats(completed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Leaderboard CSV
    rows = build_leaderboard_rows(completed)
    write_leaderboard_csv(rows, output_dir / "leaderboard.csv")

    # Summary markdown
    md = build_summary_md(experiment.name, experiment.shot_id, completed, total_cost, phase_label)
    (output_dir / "summary.md").write_text(md)


if __name__ == "__main__":
    asyncio.run(main())
