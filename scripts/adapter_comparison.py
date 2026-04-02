"""Adapter vs reference-only comparison test.

Ingests a product, trains a LoRA adapter, renders with both strategies,
and compares sanity filter pass rate + deep QA scores.

Results saved to data/results/<timestamp>_adapter-comparison/

Run with: uv run python scripts/adapter_comparison.py
Requires FAL_KEY and GEMINI_API_KEY in .env
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    from product_fidelity_lab.config import get_settings
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.generation.presets import BUILTIN_PRESETS
    from product_fidelity_lab.models.product import PackagingType
    from product_fidelity_lab.models.run import RunType
    from product_fidelity_lab.storage.product_store import ProductStore
    from product_fidelity_lab.storage.results import ResultsWriter
    from product_fidelity_lab.storage.run_store import RunStore

    settings = get_settings()
    if not settings.live_ready:
        print("ERROR: Set FAL_KEY and GEMINI_API_KEY in .env")
        sys.exit(1)

    results = ResultsWriter("adapter-comparison")
    print("=" * 60)
    print("ADAPTER vs REFERENCE-ONLY COMPARISON")
    print(f"Results: {results.dir}")
    print("=" * 60)

    # Setup
    db_path = Path("data/adapter_cmp.db")
    products_dir = Path("data/adapter_cmp_products")
    runs_dir = Path("data/adapter_cmp_runs")

    run_store = RunStore(db_path=db_path, runs_dir=runs_dir)
    await run_store.initialize()
    product_store = ProductStore(db_path=db_path, products_dir=products_dir)
    await product_store.initialize()
    await product_store.seed_builtin_presets(BUILTIN_PRESETS)

    fal = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )

    # ── 1. Ingest ────────────────────────────────────────────────
    print("\n1. INGEST")
    product = await product_store.create_product("Bacardi Superior")
    photos = [
        Path("data/golden/training/hero_front_straight.jpg"),
        Path("data/golden/training/label_closeup.jpg"),
        Path("data/golden/training/hero_front_left_45.jpg"),
        Path("data/golden/training/hero_front_right_45.jpg"),
    ]
    for p in photos:
        if not p.exists():
            print(f"   ERROR: {p} not found")
            sys.exit(1)

    from product_fidelity_lab.product.ingest import ingest_product_images

    ingest_run = await run_store.create_run(
        RunType.INGEST, config={"product_id": product.id},
    )
    profile = await ingest_product_images(
        product.id,
        photos,
        product_store=product_store,
        fal_client=fal,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        run_store=run_store,
        run_id=ingest_run.id,
    )
    print(f"   Product: {product.id}")
    print(f"   Confidence: {profile.ingest_confidence}")
    print(f"   Views: {[v.value for v in profile.views.keys()]}")

    # Save reference images
    for p in photos:
        await results.save_image(
            f"file://{p.resolve()}", p.stem, subdir="references",
        )

    # Confirm
    profile.critical_texts = ["BACARDI"]
    profile.packaging_type = PackagingType.BOTTLE
    await product_store.update_product(product.id, profile=profile)

    results.set_metadata("product", {
        "id": product.id,
        "name": "Bacardi Superior",
        "confidence": profile.ingest_confidence,
        "views": [v.value for v in profile.views.keys()],
        "critical_texts": profile.critical_texts,
        "packaging": profile.packaging_type.value,
    })

    # ── 2. Train adapter ─────────────────────────────────────────
    print("\n2. TRAIN ADAPTER")
    from product_fidelity_lab.models.product import AssetType
    from product_fidelity_lab.product.provider import FalLoraProvider

    assets = await product_store.get_assets(product.id)
    train_urls = [
        a.fal_url
        for a in assets
        if a.asset_type == AssetType.RAW_UPLOAD and a.fal_url
    ]
    print(f"   Training on {len(train_urls)} images...")

    train_fal = FalClient(timeout_s=600, max_concurrent=1)
    provider = FalLoraProvider(train_fal)

    train_start = time.monotonic()
    model = await provider.start_training(
        product.id, train_urls, "BACARDIBOTTLE", product_store,
    )
    train_ms = int((time.monotonic() - train_start) * 1000)
    print(f"   Status: {model.status.value}")
    print(f"   Duration: {train_ms}ms")
    if model.external_model_id:
        print(f"   Model: {model.external_model_id[:80]}...")

    results.set_metadata("training", {
        "status": model.status.value,
        "duration_ms": train_ms,
        "images": len(train_urls),
        "model_id": model.external_model_id,
    })

    # ── 3. Render: reference-only ────────────────────────────────
    print("\n3. RENDER: REFERENCE-ONLY")
    ref_results = await _render_and_evaluate(
        product, profile, product_store, run_store, fal, settings, results,
        strategy_label="reference_only",
        product_model=None,
        seed=200,
    )

    # ── 4. Render: adapter ───────────────────────────────────────
    adapter_model = await product_store.get_model(product.id)
    if adapter_model and adapter_model.external_model_id:
        print("\n4. RENDER: ADAPTER")
        adapter_results = await _render_and_evaluate(
            product, profile, product_store, run_store, fal, settings, results,
            strategy_label="adapter",
            product_model=adapter_model,
            seed=200,
        )
    else:
        print("\n4. RENDER: ADAPTER — skipped (training failed)")
        adapter_results = None

    # ── 5. Comparison ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    def _fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(f"\n{'':30s} {'Reference':>12s} {'Adapter':>12s}")
    print("-" * 55)

    rows = [
        ("Filter pass rate", ref_results["filter_pass_rate"],
         adapter_results["filter_pass_rate"] if adapter_results else "N/A"),
        ("QA overall", ref_results["qa_overall"],
         adapter_results["qa_overall"] if adapter_results else "N/A"),
        ("QA grade", ref_results["qa_grade"],
         adapter_results["qa_grade"] if adapter_results else "N/A"),
        ("QA passed", ref_results["qa_passed"],
         adapter_results["qa_passed"] if adapter_results else "N/A"),
        ("AFV score", ref_results["afv_score"],
         adapter_results["afv_score"] if adapter_results else "N/A"),
        ("Brand score", ref_results["brand_score"],
         adapter_results["brand_score"] if adapter_results else "N/A"),
    ]
    for label, ref_val, ada_val in rows:
        print(f"  {label:30s} {_fmt(ref_val):>12s} {_fmt(ada_val):>12s}")

    results.set_metadata("comparison", {
        "reference_only": ref_results,
        "adapter": adapter_results,
    })
    manifest = results.write_manifest()

    print(f"\n   Results: {results.dir}")
    print(f"   Manifest: {manifest}")

    # Cleanup temp data (keep results)
    db_path.unlink(missing_ok=True)
    shutil.rmtree(products_dir, ignore_errors=True)
    shutil.rmtree(runs_dir, ignore_errors=True)


async def _render_and_evaluate(
    product,
    profile,
    product_store,
    run_store,
    fal,
    settings,
    results: "ResultsWriter",
    strategy_label: str,
    product_model,
    seed: int,
) -> dict:
    """Render, filter, rank, repair, evaluate, save all images."""
    from product_fidelity_lab.critic.fast_ranker import (
        build_filter_summary,
        gemini_rank,
        sanity_filter,
    )
    from product_fidelity_lab.evaluation.product_eval import run_product_evaluation
    from product_fidelity_lab.generation.presets import BUILTIN_PRESETS, compile_prompt
    from product_fidelity_lab.generation.strategy import route_strategy
    from product_fidelity_lab.repair.composite import (
        execute_repair,
        is_repair_eligible,
        plan_repair,
    )
    from product_fidelity_lab.storage.product_store import is_usable_reference

    preset = BUILTIN_PRESETS["clean_packshot"]
    strategy = route_strategy(product, preset, product_model)
    print(f"   Strategy: {type(strategy).__name__}")

    assets = await product_store.get_assets(product.id)
    ref_assets = [a for a in assets if is_usable_reference(a) and a.fal_url]
    ref_urls = [a.fal_url for a in ref_assets if a.fal_url]

    # Generate
    candidates = await strategy.generate_candidates(
        profile,
        preset,
        ref_urls,
        num_candidates=2,
        fal_client=fal,
        base_seed=seed,
    )
    print(f"   Generated {len(candidates)} candidates")

    # Save candidate images
    for c in candidates:
        try:
            await results.save_image(
                c.image_url, f"seed{c.seed}", subdir=strategy_label,
            )
        except Exception:
            print(f"   Failed to save candidate seed={c.seed}")

    # Filter
    await sanity_filter(candidates, profile, preset, fal_client=fal)
    f_summary = build_filter_summary(candidates)
    passed = [c for c in candidates if c.filter_passed]
    failed = [c for c in candidates if not c.filter_passed]

    if not passed and failed:
        least_bad = min(failed, key=lambda c: len(c.filter_reasons))
        passed = [least_bad]
        print(f"   ALL FILTERED — fallback to {least_bad.candidate_id}")

    filter_pass_rate = (
        f_summary["passed"] / f_summary["total"] if f_summary["total"] else 0
    )
    print(f"   Filter: {f_summary}")

    # Rank
    if len(passed) > 1:
        prompt = compile_prompt(preset, profile, len(ref_urls))
        await gemini_rank(
            passed, profile, preset,
            reference_urls=ref_urls,
            render_prompt=prompt,
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_model,
        )

    selected = passed[:1]

    # Repair
    repaired = []
    if is_repair_eligible(preset) and selected:
        for sel in selected:
            rp = await plan_repair(sel, profile, preset)
            if rp.actions:
                try:
                    fixed = await execute_repair(sel, rp, product_store, fal)
                    repaired.append(fixed)
                except Exception:
                    repaired.append(sel)
            else:
                repaired.append(sel)
    else:
        repaired = list(selected)

    final_url = repaired[0].image_url if repaired else (
        selected[0].image_url if selected else None
    )

    # Save repaired
    if final_url:
        try:
            await results.save_image(
                final_url, "repaired", subdir=strategy_label,
            )
        except Exception:
            pass

    # Evaluate
    qa_result = {
        "overall": 0, "grade": "F", "passed": False,
        "afv": {"score": 0}, "brand": {"combined_score": 0},
    }
    if final_url:
        qa_result = await run_product_evaluation(
            final_url, profile,
            fal_client=fal,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
        )
        print(
            f"   QA: {qa_result['overall']:.3f} ({qa_result['grade']}) "
            f"passed={qa_result['passed']}"
        )

    return {
        "strategy": strategy_label,
        "filter_pass_rate": filter_pass_rate,
        "qa_overall": qa_result["overall"],
        "qa_grade": qa_result["grade"],
        "qa_passed": qa_result["passed"],
        "afv_score": qa_result["afv"]["score"],
        "brand_score": qa_result["brand"]["combined_score"],
        "final_url": final_url,
        "candidates": [
            {"seed": c.seed, "url": c.image_url, "filter_passed": c.filter_passed}
            for c in candidates
        ],
    }


if __name__ == "__main__":
    asyncio.run(main())
