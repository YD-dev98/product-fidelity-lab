"""Generation API endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.generation.cold_start import cold_generate
from product_fidelity_lab.generation.prompts import build_generation_prompt, select_reference_urls
from product_fidelity_lab.main import get_run_store
from product_fidelity_lab.models.generation import GenerationRequest
from product_fidelity_lab.models.run import RunStatus, RunType

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate", status_code=202)
async def generate(
    req: GenerationRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Start a generation job with a raw prompt. Returns run_id."""
    settings = get_settings()
    if settings.replay_mode or not settings.live_ready:
        raise HTTPException(503, "Live generation unavailable — server is in replay mode")
    store = get_run_store()
    run = await store.create_run(RunType.GENERATION, config=req.model_dump())

    background_tasks.add_task(_run_generation_job, run.id, req)
    return {"run_id": run.id}


class GenerateForSpecRequest(BaseModel):
    spec_id: str
    seed: int | None = None


@router.post("/generate/for-spec", status_code=202)
async def generate_for_spec(
    req: GenerateForSpecRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Generate an image conditioned on a golden spec.

    Builds a shot-aware prompt and selects multi-reference images
    automatically from the golden set.
    """
    settings = get_settings()
    if settings.replay_mode or not settings.live_ready:
        raise HTTPException(503, "Live generation unavailable — server is in replay mode")

    specs_dir = settings.data_dir / "golden" / "specs"
    all_specs = load_all_specs(specs_dir)
    spec = next((s for s in all_specs if s.shot_id == req.spec_id), None)
    if spec is None:
        raise HTTPException(404, f"Spec not found: {req.spec_id}")

    ref_urls = select_reference_urls(spec, all_specs, max_refs=4)
    prompt = build_generation_prompt(spec, n_references=len(ref_urls))

    gen_req = GenerationRequest(
        prompt=prompt,
        reference_urls=ref_urls,
        seed=req.seed,
    )

    store = get_run_store()
    run = await store.create_run(
        RunType.GENERATION,
        config={**gen_req.model_dump(), "spec_id": req.spec_id},
    )

    background_tasks.add_task(_run_generation_job, run.id, gen_req)
    return {"run_id": run.id}


async def _run_generation_job(
    run_id: str,
    req: GenerationRequest,
) -> None:
    """Background task for generation."""
    settings = get_settings()
    store = get_run_store()

    try:
        await store.update_status(run_id, RunStatus.RUNNING)

        fal = FalClient(
            timeout_s=settings.fal_timeout_s,
            max_concurrent=settings.fal_max_concurrent,
        )

        from product_fidelity_lab.storage.result_cache import ResultCache

        gen_cache = ResultCache(settings.data_dir / "cache" / "results" / "generation")
        result = await cold_generate(req, fal, cache=gen_cache)

        result_dict: dict[str, Any] = result.model_dump()
        await store.update_result(
            run_id,
            result=result_dict,
            duration_ms=result.duration_ms,
            cost=result.cost_estimate,
        )
        await store.update_status(run_id, RunStatus.COMPLETE)

    except Exception as exc:
        logger.error("generation.job_failed", run_id=run_id, error=str(exc))
        await store.update_status(run_id, RunStatus.FAILED)
