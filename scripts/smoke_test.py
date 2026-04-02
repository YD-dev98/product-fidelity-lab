"""End-to-end smoke test: ingest → profile → render → rank.

Run with: uv run python scripts/smoke_test.py

Requires FAL_KEY and GEMINI_API_KEY in .env
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    from product_fidelity_lab.config import get_settings
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.generation.presets import BUILTIN_PRESETS
    from product_fidelity_lab.models.run import RunType
    from product_fidelity_lab.product.ingest import ingest_product_images
    from product_fidelity_lab.storage.product_store import ProductStore
    from product_fidelity_lab.storage.run_store import RunStore

    settings = get_settings()
    if not settings.live_ready:
        print("ERROR: API keys not configured. Set FAL_KEY and GEMINI_API_KEY in .env")
        sys.exit(1)

    print("=== Smoke Test: Full Pipeline ===\n")

    # Init stores
    db_path = Path("data/smoke_test.db")
    products_dir = Path("data/smoke_products")
    runs_dir = Path("data/smoke_runs")

    run_store = RunStore(db_path=db_path, runs_dir=runs_dir)
    await run_store.initialize()

    product_store = ProductStore(db_path=db_path, products_dir=products_dir)
    await product_store.initialize()
    await product_store.seed_builtin_presets(BUILTIN_PRESETS)

    fal = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )

    # Use 2 existing product photos
    photos = [
        Path("data/golden/training/hero_front_straight.jpg"),
        Path("data/golden/training/label_closeup.jpg"),
    ]
    for p in photos:
        if not p.exists():
            print(f"ERROR: Photo not found: {p}")
            sys.exit(1)

    # Step 1: Create product
    print("1. Creating product...")
    product = await product_store.create_product("Smoke Test Product")
    print(f"   Product ID: {product.id}")

    # Step 2: Ingest
    print("\n2. Ingesting images...")
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
    print(f"   Confidence: {profile.ingest_confidence}")
    print(f"   Brand texts: {profile.brand_texts[:5]}")
    print(f"   Colors: {profile.brand_colors_hex[:3]}")
    print(f"   Views: {list(profile.views.keys())}")
    print(f"   Logo asset: {profile.logo_asset_id}")
    print(f"   Label asset: {profile.label_asset_id}")
    print(f"   Alpha mask: {profile.alpha_mask_asset_id}")
    print(f"   Packaging: {profile.packaging_type}")

    # Step 3: List assets
    assets = await product_store.get_assets(product.id)
    print(f"\n   Total assets: {len(assets)}")
    for a in assets:
        print(f"   - {a.asset_type.value}: {a.angle_tag.value} ({a.id})")

    # Step 4: Render with clean_packshot (2 candidates to keep cost low)
    print("\n3. Rendering with clean_packshot preset (2 candidates)...")
    from product_fidelity_lab.generation.presets import compile_prompt
    from product_fidelity_lab.generation.strategy import route_strategy
    from product_fidelity_lab.storage.product_store import is_usable_reference

    preset = BUILTIN_PRESETS["clean_packshot"]

    ref_assets = [
        a for a in assets if is_usable_reference(a) and a.fal_url
    ]
    ref_urls = [a.fal_url for a in ref_assets if a.fal_url]
    print(f"   Using {len(ref_urls)} reference URLs")

    strategy = route_strategy(product, preset)
    candidates = await strategy.generate_candidates(
        profile, preset, ref_urls,
        num_candidates=2,
        fal_client=fal,
        base_seed=42,
    )
    print(f"   Generated {len(candidates)} candidates")
    for c in candidates:
        print(f"   - seed={c.seed} url={c.image_url[:80]}...")

    # Step 5: Sanity filter
    print("\n4. Running sanity filters...")
    from product_fidelity_lab.critic.fast_ranker import (
        build_filter_summary,
        sanity_filter,
    )

    await sanity_filter(candidates, profile, preset, fal_client=fal)
    summary = build_filter_summary(candidates)
    print(f"   Filter result: {summary}")
    for c in candidates:
        status = "PASS" if c.filter_passed else f"FAIL ({c.filter_reasons})"
        print(f"   - {c.candidate_id}: {status}")

    # Step 6: Gemini ranking (only if >1 passed)
    passed = [c for c in candidates if c.filter_passed]
    if len(passed) > 1:
        print("\n5. Running Gemini ranking...")
        from product_fidelity_lab.critic.fast_ranker import gemini_rank

        prompt = compile_prompt(preset, profile, len(ref_urls))
        await gemini_rank(
            passed, profile, preset,
            reference_urls=ref_urls,
            render_prompt=prompt,
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_model,
        )
        for c in passed:
            print(
                f"   - {c.candidate_id}: rank={c.rank_score} "
                f"id={c.identity_score} label={c.label_score} "
                f"style={c.style_score} comp={c.composition_score}"
            )
    else:
        print("\n5. Skipping Gemini ranking (<=1 passed filter)")

    # Step 7: Repair planning
    print("\n6. Planning repair...")
    from product_fidelity_lab.repair.composite import is_repair_eligible, plan_repair

    if is_repair_eligible(preset) and passed:
        best = passed[0]
        plan = await plan_repair(best, profile, preset)
        print(f"   Actions: {[a.value for a in plan.actions]}")
        if plan.logo_asset_id:
            print(f"   Logo asset: {plan.logo_asset_id}")
        if plan.label_asset_id:
            print(f"   Label asset: {plan.label_asset_id}")
        if plan.edit_prompt:
            print(f"   Edit prompt: {plan.edit_prompt[:80]}")
    else:
        print("   Repair not eligible for this preset/results")

    print("\n=== Smoke Test Complete ===")

    # Cleanup
    import shutil

    db_path.unlink(missing_ok=True)
    shutil.rmtree(products_dir, ignore_errors=True)
    shutil.rmtree(runs_dir, ignore_errors=True)
    print("Cleaned up smoke test data.")


if __name__ == "__main__":
    asyncio.run(main())
