"""Render API endpoints."""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.main import get_product_store, get_run_store
from product_fidelity_lab.models.preset import (
    Candidate,
    QAResult,
    RenderRequest,
    RenderResult,
    RenderTrace,
    RepairAction,
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

        # Route strategy (use adapter if product has a trained model)
        from product_fidelity_lab.generation.strategy import route_strategy

        product_model = await product_store.get_model(req.product_id)
        strategy = route_strategy(product, preset, product_model)

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

        # ── Repair ──────────────────────────────────────────────────
        from product_fidelity_lab.repair.composite import (
            execute_repair,
            is_repair_eligible,
            plan_repair,
        )

        repaired: list[Candidate] = []
        repair_actions: list[RepairAction] = []

        if (
            not req.skip_repair
            and is_repair_eligible(preset)
            and selected
        ):
            for sel in selected:
                plan = await plan_repair(sel, profile, preset)
                if plan.actions:
                    fixed = await execute_repair(
                        sel, plan, product_store, fal,
                    )
                    repaired.append(fixed)
                    repair_actions.extend(plan.actions)
                else:
                    repaired.append(sel)
        else:
            repaired = list(selected)

        # ── Build trace ─────────────────────────────────────────────
        trace = RenderTrace(
            final_prompt=final_prompt,
            selected_reference_asset_ids=reference_asset_ids,
            selected_reference_urls=reference_urls,
            provider_payload=provider_payload,
            strategy_used=candidates[0].strategy_used if candidates else "reference_only",
            seeds=[c.seed for c in candidates],
            repair_actions=repair_actions,
            preset_snapshot=preset,
            profile_snapshot=profile,
            ranker_version=ranker_version,
            filter_summary=filter_summary,
            judge_summary=judge_summary,
        )

        # ── QA evaluation ───────────────────────────────────────────
        final_url = repaired[0].image_url if repaired else (
            selected[0].image_url if selected else None
        )
        qa_result: QAResult | None = None

        if not req.skip_qa and final_url and profile.critical_texts:
            try:
                qa_result = await _run_qa(
                    final_url, profile, fal,
                )
            except Exception:
                logger.warning("render.qa_failed", run_id=run_id)

        # ── Build result ────────────────────────────────────────────
        result = RenderResult(
            run_id=run_id,
            product_id=req.product_id,
            preset_id=req.preset_id,
            candidates=candidates,
            selected=selected,
            repaired=repaired,
            final_image_url=final_url,
            total_duration_ms=total_ms,
            total_cost=total_cost,
            trace=trace,
            warnings=warnings,
            qa=qa_result,
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


async def _run_qa(
    image_url: str,
    profile: Any,
    fal_client: Any,
) -> QAResult:
    """Run lightweight brand QA on the final rendered image.

    Uses the existing OCR + color evaluation infrastructure against
    the product profile (not a golden spec).
    """
    from product_fidelity_lab.evaluation.layer_brand import (
        compare_text,
        extract_text,
    )
    from product_fidelity_lab.models.golden_spec import ExpectedText

    # Build expected texts from profile critical_texts
    expected = [
        ExpectedText(text=t, critical=True, match_mode="exact_token")
        for t in profile.critical_texts
    ]

    # Run OCR
    extracted = await extract_text(image_url, fal_client)
    text_result = compare_text(expected, extracted)

    # Run color comparison
    color_score = 1.0
    if profile.brand_colors_hex:
        from io import BytesIO

        from PIL import Image

        from product_fidelity_lab.evaluation.color import (
            compare_to_brand_colors,
            extract_dominant_colors,
        )
        from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

        img_bytes = await fetch_image_bytes(image_url)
        img = Image.open(BytesIO(img_bytes))
        extracted_lab = extract_dominant_colors(img, n_colors=5)
        color_score, _ = compare_to_brand_colors(extracted_lab, profile.brand_colors_hex)

    combined = 0.4 * text_result.score + 0.6 * color_score
    passed = combined >= 0.7 and not text_result.critical_failures

    logger.info(
        "render.qa_complete",
        text_score=text_result.score,
        color_score=color_score,
        combined=combined,
        passed=passed,
    )

    return QAResult(
        text_score=text_result.score,
        color_score=color_score,
        combined_score=round(combined, 3),
        critical_text_failures=text_result.critical_failures,
        passed=passed,
    )
