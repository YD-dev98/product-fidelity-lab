"""Render API endpoints."""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.main import get_product_store, get_run_store
from product_fidelity_lab.models.preset import (
    RenderRequest,
    RenderResult,
    RenderTrace,
    StudioPreset,
)
from product_fidelity_lab.models.run import RunStatus, RunType
from product_fidelity_lab.product.validation import validate_compatibility
from product_fidelity_lab.storage.product_store import is_usable_reference

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["render"])


@router.post("/render", status_code=202)
async def render(
    req: RenderRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Start a render job. Returns run_id immediately."""
    settings = get_settings()
    if not settings.live_ready:
        raise HTTPException(503, "Live mode unavailable — API keys not configured")

    product_store = get_product_store()
    run_store = get_run_store()

    # Load product + profile
    product = await product_store.get_product(req.product_id)
    if product is None:
        raise HTTPException(404, "Product not found")

    profile = await product_store.get_profile(req.product_id)
    if profile is None:
        raise HTTPException(422, "Product has no profile — upload images first")

    # Load preset
    preset = await product_store.get_preset(req.preset_id)
    if preset is None:
        raise HTTPException(404, f"Preset not found: {req.preset_id}")

    # Validate preset-product compatibility
    warnings = await validate_compatibility(profile, preset, product_store, req.product_id)
    errors = [w for w in warnings if w.startswith("ERROR:")]
    if errors:
        raise HTTPException(422, {"errors": errors})

    run = await run_store.create_run(
        RunType.RENDER,
        config={
            "product_id": req.product_id,
            "preset_id": req.preset_id,
            "num_candidates": req.num_candidates,
        },
    )

    background_tasks.add_task(
        _run_render_job, run.id, req, [w for w in warnings if not w.startswith("ERROR:")]
    )

    return {"run_id": run.id}


@router.get("/render/{run_id}")
async def get_render(run_id: str) -> dict[str, Any]:
    """Get render job status and result. Uses Run as status envelope."""
    run_store = get_run_store()
    run = await run_store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run.type != RunType.RENDER:
        raise HTTPException(404, "Not a render run")
    return run.model_dump()


@router.get("/presets")
async def list_presets() -> list[dict[str, Any]]:
    store = get_product_store()
    presets = await store.list_presets()
    return [p.model_dump() for p in presets]


@router.post("/presets", status_code=201)
async def create_preset(preset: StudioPreset) -> dict[str, Any]:
    store = get_product_store()
    await store.save_preset(preset)
    return preset.model_dump()


async def _run_render_job(
    run_id: str,
    req: RenderRequest,
    warnings: list[str],
) -> None:
    """Background task for rendering."""
    settings = get_settings()
    product_store = get_product_store()
    run_store = get_run_store()

    try:
        await run_store.update_status(run_id, RunStatus.RUNNING)

        # Load everything
        product = await product_store.get_product(req.product_id)
        if product is None:
            raise ValueError("Product not found")

        profile = await product_store.get_profile(req.product_id)
        if profile is None:
            raise ValueError("Profile not found")

        preset = await product_store.get_preset(req.preset_id)
        if preset is None:
            raise ValueError("Preset not found")

        # Gather all usable refs for generation (angle filtering is a separate concern)
        assets = await product_store.get_assets(req.product_id)
        reference_assets = [
            a for a in assets
            if is_usable_reference(a) and a.fal_url
        ]

        reference_urls = [a.fal_url for a in reference_assets if a.fal_url]
        reference_asset_ids = [a.id for a in reference_assets if a.fal_url]

        if not reference_urls:
            raise ValueError("No usable reference images with URLs")

        # Route strategy
        from product_fidelity_lab.generation.strategy import route_strategy

        strategy = route_strategy(product, preset)

        # Generate candidates
        from product_fidelity_lab.generation.client import FalClient

        fal = FalClient(
            timeout_s=settings.fal_timeout_s,
            max_concurrent=settings.fal_max_concurrent,
        )

        effective_aspect = req.aspect_ratio or preset.aspect_ratio

        start = time.monotonic()
        candidates = await strategy.generate_candidates(
            profile,
            preset,
            reference_urls,
            req.num_candidates,
            fal,
            style_prompt=req.style_prompt,
            base_seed=req.seed,
            aspect_ratio_override=req.aspect_ratio,
        )
        total_ms = int((time.monotonic() - start) * 1000)
        total_cost = sum(c.cost_estimate for c in candidates)

        # Build sanitized provider payload
        from product_fidelity_lab.generation.presets import compile_prompt

        final_prompt = compile_prompt(
            preset, profile, len(reference_urls), req.style_prompt
        )
        provider_payload = {
            "model_id": candidates[0].model_id if candidates else "",
            "prompt": final_prompt,
            "image_refs": reference_urls,
            "seed": req.seed,
            "inference_steps": preset.num_inference_steps or 28,
            "guidance_scale": preset.guidance_scale or 3.5,
            "image_size": effective_aspect,
        }

        # ── Ranking ──────────────────────────────────────────────────
        from product_fidelity_lab.critic.fast_ranker import (
            RANKER_VERSION,
            build_filter_summary,
            build_judge_summary,
            gemini_rank,
            sanity_filter,
        )

        ranker_version: str | None = None
        filter_summary: dict[str, object] = {}
        judge_summary: dict[str, object] = {}

        if not req.skip_ranking and candidates:
            ranker_version = RANKER_VERSION

            # Stage 1: sanity filters (pass fal for OCR check on text-critical)
            await sanity_filter(candidates, profile, preset, fal_client=fal)
            filter_summary = build_filter_summary(candidates)

            passed = [c for c in candidates if c.filter_passed]
            failed = [c for c in candidates if not c.filter_passed]

            if not passed and failed:
                # All filtered — fallback: keep least-bad with warning
                least_bad = min(failed, key=lambda c: len(c.filter_reasons))
                passed = [least_bad]
                warnings.append(
                    f"All candidates failed sanity filters — "
                    f"using fallback (seed={least_bad.seed})"
                )

            # Stage 2: Gemini scoring on survivors
            if len(passed) > 1:
                await gemini_rank(
                    passed,
                    profile,
                    preset,
                    reference_urls=reference_urls,
                    render_prompt=final_prompt,
                    api_key=settings.gemini_api_key,
                    model_id=settings.gemini_model,
                )
                judge_summary = build_judge_summary(passed, settings.gemini_model)

            selected = passed[:3]
        else:
            selected = candidates

        # ── Build trace ─────────────────────────────────────────────
        trace = RenderTrace(
            final_prompt=final_prompt,
            selected_reference_asset_ids=reference_asset_ids,
            selected_reference_urls=reference_urls,
            provider_payload=provider_payload,
            strategy_used=candidates[0].strategy_used if candidates else "reference_only",
            seeds=[c.seed for c in candidates],
            preset_snapshot=preset,
            profile_snapshot=profile,
            ranker_version=ranker_version,
            filter_summary=filter_summary,
            judge_summary=judge_summary,
        )

        # ── Build result ────────────────────────────────────────────
        result = RenderResult(
            run_id=run_id,
            product_id=req.product_id,
            preset_id=req.preset_id,
            candidates=candidates,
            selected=selected,
            final_image_url=selected[0].image_url if selected else None,
            total_duration_ms=total_ms,
            total_cost=total_cost,
            trace=trace,
            warnings=warnings,
        )

        await run_store.update_status(run_id, RunStatus.COMPLETE)
        await run_store.update_result(
            run_id,
            result=result.model_dump(mode="json"),
            duration_ms=total_ms,
            cost=total_cost,
        )

        logger.info(
            "render.complete",
            run_id=run_id,
            candidates=len(candidates),
            duration_ms=total_ms,
        )

    except Exception as exc:
        logger.error("render.job_failed", run_id=run_id, error=str(exc))
        await run_store.update_status(run_id, RunStatus.FAILED)
