"""Full end-to-end smoke test: ingest → confirm → render → rank → repair → evaluate.

Run with: uv run python scripts/smoke_test_full.py

Requires FAL_KEY and GEMINI_API_KEY in .env
"""

from __future__ import annotations

import asyncio
import json
import sys
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
    from product_fidelity_lab.storage.run_store import RunStore

    settings = get_settings()
    if not settings.live_ready:
        print("ERROR: Set FAL_KEY and GEMINI_API_KEY in .env")
        sys.exit(1)

    print("=" * 60)
    print("FULL END-TO-END SMOKE TEST")
    print("=" * 60)

    # Init stores
    db_path = Path("data/smoke_full.db")
    products_dir = Path("data/smoke_full_products")
    runs_dir = Path("data/smoke_full_runs")

    run_store = RunStore(db_path=db_path, runs_dir=runs_dir)
    await run_store.initialize()

    product_store = ProductStore(db_path=db_path, products_dir=products_dir)
    await product_store.initialize()
    await product_store.seed_builtin_presets(BUILTIN_PRESETS)

    fal = FalClient(timeout_s=settings.fal_timeout_s, max_concurrent=settings.fal_max_concurrent)

    # ── 1. Create product ────────────────────────────────────────
    print("\n1. CREATE PRODUCT")
    product = await product_store.create_product("Bacardi Superior")
    print(f"   ID: {product.id}")

    # ── 2. Upload 3 images ───────────────────────────────────────
    print("\n2. UPLOAD + INGEST (3 images)")
    photos = [
        Path("data/golden/training/hero_front_straight.jpg"),
        Path("data/golden/training/label_closeup.jpg"),
        Path("data/golden/training/hero_front_left_45.jpg"),
    ]
    for p in photos:
        if not p.exists():
            print(f"   ERROR: {p} not found")
            sys.exit(1)
        print(f"   - {p.name}")

    from product_fidelity_lab.product.ingest import ingest_product_images

    ingest_run = await run_store.create_run(RunType.INGEST, config={"product_id": product.id})
    profile = await ingest_product_images(
        product.id, photos,
        product_store=product_store, fal_client=fal,
        gemini_api_key=settings.gemini_api_key, gemini_model=settings.gemini_model,
        run_store=run_store, run_id=ingest_run.id,
    )

    print(f"   Confidence: {profile.ingest_confidence}")
    print(f"   Brand texts: {profile.brand_texts[:3]}")
    print(f"   Colors: {profile.brand_colors_hex[:3]}")
    print(f"   Views: {[v.value for v in profile.views.keys()]}")
    print(f"   Logo asset: {profile.logo_asset_id}")
    print(f"   Label asset: {profile.label_asset_id}")
    print(f"   Packaging: {profile.packaging_type.value}")

    assets = await product_store.get_assets(product.id)
    print(f"   Total assets: {len(assets)}")
    for a in assets:
        bbox = a.metadata.get("bbox", "")
        extra = f" bbox={bbox}" if bbox else ""
        print(f"     {a.asset_type.value:12s} {a.angle_tag.value:16s} {a.id}{extra}")

    # ── 3. Confirm profile ───────────────────────────────────────
    print("\n3. CONFIRM PROFILE")
    profile.critical_texts = ["BACARDI"]
    profile.packaging_type = PackagingType.BOTTLE
    await product_store.update_product(product.id, profile=profile)
    print(f"   Critical texts: {profile.critical_texts}")
    print(f"   Packaging: {profile.packaging_type.value}")

    # ── 4. Render ────────────────────────────────────────────────
    print("\n4. RENDER (clean_packshot, 2 candidates)")
    from product_fidelity_lab.critic.fast_ranker import (
        RANKER_VERSION,
        build_filter_summary,
        build_judge_summary,
        gemini_rank,
        sanity_filter,
    )
    from product_fidelity_lab.generation.presets import compile_prompt
    from product_fidelity_lab.generation.strategy import route_strategy
    from product_fidelity_lab.repair.composite import execute_repair, is_repair_eligible, plan_repair
    from product_fidelity_lab.storage.product_store import is_usable_reference

    preset = BUILTIN_PRESETS["clean_packshot"]
    product_model = await product_store.get_model(product.id)
    strategy = route_strategy(product, preset, product_model)
    print(f"   Strategy: {type(strategy).__name__}")

    ref_assets = [a for a in assets if is_usable_reference(a) and a.fal_url]
    ref_urls = [a.fal_url for a in ref_assets if a.fal_url]
    print(f"   References: {len(ref_urls)}")

    import time

    start = time.monotonic()
    candidates = await strategy.generate_candidates(
        profile, preset, ref_urls, num_candidates=2,
        fal_client=fal, base_seed=100,
    )
    gen_ms = int((time.monotonic() - start) * 1000)
    print(f"   Generated {len(candidates)} candidates in {gen_ms}ms")
    for c in candidates:
        print(f"     seed={c.seed} {c.image_url[:70]}...")

    # ── 5. Sanity filter ─────────────────────────────────────────
    print("\n5. SANITY FILTER")
    await sanity_filter(candidates, profile, preset, fal_client=fal)
    f_summary = build_filter_summary(candidates)
    print(f"   {f_summary}")
    for c in candidates:
        status = "PASS" if c.filter_passed else f"FAIL {c.filter_reasons}"
        print(f"     {c.candidate_id}: {status}")

    # ── 6. Gemini rank ───────────────────────────────────────────
    passed = [c for c in candidates if c.filter_passed]
    failed = [c for c in candidates if not c.filter_passed]

    # Fallback: if all filtered, keep the least-bad
    if not passed and failed:
        least_bad = min(failed, key=lambda c: len(c.filter_reasons))
        passed = [least_bad]
        print(f"   ALL FILTERED — fallback to {least_bad.candidate_id}")

    if len(passed) > 1:
        print("\n6. GEMINI RANK")
        prompt = compile_prompt(preset, profile, len(ref_urls))
        await gemini_rank(
            passed, profile, preset,
            reference_urls=ref_urls, render_prompt=prompt,
            api_key=settings.gemini_api_key, model_id=settings.gemini_model,
        )
        j_summary = build_judge_summary(passed, settings.gemini_model)
        print(f"   {j_summary}")
        for c in passed:
            print(
                f"     {c.candidate_id}: rank={c.rank_score} "
                f"id={c.identity_score} label={c.label_score} "
                f"style={c.style_score} comp={c.composition_score}"
            )
        selected = passed[:3]
    else:
        print("\n6. GEMINI RANK — skipped (<=1 passed)")
        selected = passed

    # ── 7. Repair ────────────────────────────────────────────────
    print("\n7. REPAIR")
    repaired = []
    if is_repair_eligible(preset) and selected:
        for sel in selected:
            rp = await plan_repair(sel, profile, preset)
            print(f"     {sel.candidate_id}: actions={[a.value for a in rp.actions]}")
            if rp.actions:
                try:
                    fixed = await execute_repair(sel, rp, product_store, fal)
                    repaired.append(fixed)
                    print(f"       repaired → {fixed.image_url[:70]}...")
                except Exception as exc:
                    print(f"       repair failed: {exc}")
                    repaired.append(sel)
            else:
                repaired.append(sel)
    else:
        print("   Not eligible or no selected candidates")
        repaired = list(selected)

    final_url = repaired[0].image_url if repaired else (selected[0].image_url if selected else None)
    print(f"\n   FINAL IMAGE: {final_url}")

    # ── 8. Store render result ───────────────────────────────────
    print("\n8. STORE RENDER RUN")
    from product_fidelity_lab.models.preset import RenderResult, RenderTrace

    render_run = await run_store.create_run(
        RunType.RENDER,
        config={"product_id": product.id, "preset_id": "clean_packshot"},
    )
    from product_fidelity_lab.models.run import RunStatus

    await run_store.update_status(render_run.id, RunStatus.COMPLETE)
    render_result = RenderResult(
        run_id=render_run.id, product_id=product.id, preset_id="clean_packshot",
        candidates=candidates, selected=selected, repaired=repaired,
        final_image_url=final_url,
        total_duration_ms=gen_ms, total_cost=sum(c.cost_estimate for c in candidates),
    )
    await run_store.update_result(
        render_run.id, result=render_result.model_dump(mode="json"),
    )
    print(f"   Render run: {render_run.id}")

    # ── 9. Product evaluation ────────────────────────────────────
    print("\n9. PRODUCT EVALUATION (deep QA)")
    if final_url:
        from product_fidelity_lab.evaluation.product_eval import run_product_evaluation

        eval_result = await run_product_evaluation(
            final_url, profile,
            fal_client=fal,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
        )
        print(f"   Overall: {eval_result['overall']}")
        print(f"   Grade: {eval_result['grade']}")
        print(f"   Passed: {eval_result['passed']}")
        print(f"   AFV score: {eval_result['afv']['score']}")
        print(f"   Brand score: {eval_result['brand']['combined_score']}")
        print(f"   Critical failures: {eval_result['critical_failures']}")
        print(f"   Duration: {eval_result['duration_ms']}ms")
    else:
        print("   No final image to evaluate")

    # ── Cleanup ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)

    import shutil

    db_path.unlink(missing_ok=True)
    shutil.rmtree(products_dir, ignore_errors=True)
    shutil.rmtree(runs_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
