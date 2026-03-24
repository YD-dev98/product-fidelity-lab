"""Evaluation orchestrator — runs all 3 layers concurrently."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

from product_fidelity_lab.evaluation.aggregator import build_report
from product_fidelity_lab.evaluation.layer_afv import verify_facts
from product_fidelity_lab.evaluation.layer_brand import evaluate_brand
from product_fidelity_lab.evaluation.layer_depth import compare_depth, get_depth_map
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.evaluation import (
    GRADE_THRESHOLDS,
    PASS_THRESHOLD,
    AFVReport,
    BrandReport,
    DepthScore,
    EvaluationReport,
)
from product_fidelity_lab.models.golden_spec import GoldenSpec
from product_fidelity_lab.models.run import LayerState, RunStatus
from product_fidelity_lab.storage.result_cache import ResultCache
from product_fidelity_lab.storage.run_store import RunStore

logger = structlog.get_logger()


def _make_caches(cache_root: Path) -> dict[str, ResultCache]:
    """Create per-layer result caches under cache_root."""
    return {
        "afv": ResultCache(cache_root / "afv"),
        "depth": ResultCache(cache_root / "depth"),
        "brand": ResultCache(cache_root / "brand"),
    }


async def run_evaluation(
    generated_image_url: str,
    golden_spec: GoldenSpec,
    golden_depth_map: Any,  # NDArray, passed in to avoid re-computing
    *,
    fal_client: FalClient,
    gemini_api_key: str,
    gemini_model: str,
    run_store: RunStore,
    run_id: str,
    grade_thresholds: dict[str, float] | None = None,
    pass_threshold: float | None = None,
    thresholds_source: str | None = None,
    cache_root: Path | None = None,
) -> EvaluationReport:
    """Run full 3-layer evaluation concurrently."""
    start = time.monotonic()

    caches = _make_caches(cache_root) if cache_root else {}

    layer_states = {
        "afv": LayerState.RUNNING,
        "depth": LayerState.RUNNING,
        "brand": LayerState.RUNNING,
    }
    await run_store.update_status(run_id, RunStatus.RUNNING, layer_states=layer_states)

    # Pre-fetch image so all layers share cached bytes (avoids concurrent 429s)
    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    await fetch_image_bytes(generated_image_url)

    # Run all layers concurrently
    afv_task = asyncio.create_task(_run_afv(
        generated_image_url, golden_spec, gemini_api_key, gemini_model,
        run_store, run_id, cache=caches.get("afv"),
    ))
    depth_task = asyncio.create_task(_run_depth(
        generated_image_url, golden_depth_map, golden_spec, fal_client,
        run_store, run_id, cache=caches.get("depth"),
    ))
    brand_task = asyncio.create_task(_run_brand(
        generated_image_url, golden_spec, fal_client,
        run_store, run_id, cache=caches.get("brand"),
    ))

    results = await asyncio.gather(afv_task, depth_task, brand_task, return_exceptions=True)

    afv_result = _unwrap_afv(results[0], golden_spec)
    depth_result = _unwrap_depth(results[1])
    brand_result = _unwrap_brand(results[2])

    # Aggregate
    await run_store.update_status(run_id, RunStatus.AGGREGATING)
    duration_ms = int((time.monotonic() - start) * 1000)

    report = build_report(
        afv_result,
        depth_result,
        brand_result,
        grade_thresholds=grade_thresholds,
        pass_threshold=pass_threshold,
        run_metadata={
            "generated_image_url": generated_image_url,
            "golden_spec_id": golden_spec.shot_id,
            "duration_ms": duration_ms,
            "grade_thresholds": grade_thresholds or GRADE_THRESHOLDS,
            "pass_threshold": (
                pass_threshold
                if pass_threshold is not None
                else (grade_thresholds or GRADE_THRESHOLDS).get("B", PASS_THRESHOLD)
            ),
            "thresholds_source": thresholds_source or "defaults",
        },
    )

    await run_store.update_result(
        run_id,
        result=report.model_dump(),
        score=report.final.overall,
        grade=report.final.grade,
        passed=report.final.passed,
        duration_ms=duration_ms,
    )
    await run_store.update_status(run_id, RunStatus.COMPLETE)

    logger.info(
        "evaluation.complete",
        run_id=run_id,
        score=report.final.overall,
        grade=report.final.grade,
        passed=report.final.passed,
        duration_ms=duration_ms,
    )
    return report


async def _run_afv(
    image_url: str,
    spec: GoldenSpec,
    api_key: str,
    model_id: str,
    store: RunStore,
    run_id: str,
    cache: ResultCache | None = None,
) -> AFVReport:
    try:
        result = await verify_facts(
            image_url, spec.atomic_facts, api_key=api_key, model_id=model_id, cache=cache,
        )
        await _update_layer(store, run_id, "afv", LayerState.COMPLETE)
        return result
    except Exception:
        await _update_layer(store, run_id, "afv", LayerState.FAILED)
        raise


async def _run_depth(
    image_url: str,
    golden_depth: Any,
    spec: GoldenSpec,
    fal: FalClient,
    store: RunStore,
    run_id: str,
    cache: ResultCache | None = None,
) -> DepthScore:
    try:
        gen_depth = await get_depth_map(image_url, fal, cache=cache)
        roi = next((r for r in spec.rois if r.label == "product"), None)
        result = compare_depth(golden_depth, gen_depth, roi=roi)
        await _update_layer(store, run_id, "depth", LayerState.COMPLETE)
        return result
    except Exception:
        await _update_layer(store, run_id, "depth", LayerState.FAILED)
        raise


async def _run_brand(
    image_url: str,
    spec: GoldenSpec,
    fal: FalClient,
    store: RunStore,
    run_id: str,
    cache: ResultCache | None = None,
) -> BrandReport:
    try:
        result = await evaluate_brand(
            image_url,
            spec.expected_texts,
            spec.brand_colors_hex,
            spec.rois,
            fal,
            cache=cache,
        )
        await _update_layer(store, run_id, "brand", LayerState.COMPLETE)
        return result
    except Exception:
        await _update_layer(store, run_id, "brand", LayerState.FAILED)
        raise


async def _update_layer(
    store: RunStore,
    run_id: str,
    layer: str,
    state: LayerState,
) -> None:
    run = await store.get_run(run_id)
    if run is None:
        return
    layers = dict(run.layer_states)
    layers[layer] = state
    await store.update_status(run_id, run.status, layer_states=layers)


def _unwrap_afv(result: AFVReport | BaseException, spec: GoldenSpec) -> AFVReport:
    if isinstance(result, BaseException):
        logger.error("afv.failed", error=str(result))
        return AFVReport(
            facts=spec.atomic_facts,
            verdicts=[],
            score=0.0,
            category_breakdown={},
            error="AFV layer failed: " + str(result),
        )
    return result


def _unwrap_depth(result: DepthScore | BaseException) -> DepthScore:
    if isinstance(result, BaseException):
        logger.error("depth.failed", error=str(result))
        return DepthScore(
            ssim=0.0,
            correlation=0.0,
            mse=1.0,
            combined=0.0,
            error="Depth layer failed: " + str(result),
        )
    return result


def _unwrap_brand(result: BrandReport | BaseException) -> BrandReport:
    if isinstance(result, BaseException):
        logger.error("brand.failed", error=str(result))
        from product_fidelity_lab.models.evaluation import ColorScore, TextMatchScore

        return BrandReport(
            text_score=TextMatchScore(matches=[], score=0.0),
            color_score=ColorScore(
                brand_colors_hex=[], extracted_colors_hex=[], pairs=[], score=0.0
            ),
            combined_score=0.0,
            error="Brand layer failed: " + str(result),
        )
    return result
