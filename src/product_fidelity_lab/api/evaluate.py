"""Evaluation API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.evaluation.calibration import load_thresholds
from product_fidelity_lab.evaluation.engine import run_evaluation
from product_fidelity_lab.evaluation.image_fetch import ALLOWED_CONTENT_TYPES, MAX_IMAGE_BYTES
from product_fidelity_lab.evaluation.layer_depth import get_depth_map
from product_fidelity_lab.evaluation.spec_loader import load_all_specs
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.main import get_run_store
from product_fidelity_lab.models.run import RunType

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["evaluation"])


class EvaluateRequest(BaseModel):
    image_url: str
    spec_id: str


@router.post("/evaluate", status_code=202)
async def evaluate(
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Start an evaluation job. Returns run_id immediately."""
    settings = get_settings()
    if settings.replay_mode or not settings.live_ready:
        raise HTTPException(503, "Live evaluation unavailable — server is in replay mode")
    store = get_run_store()
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        raise HTTPException(
            503,
            "Calibration thresholds not found. Run calibration before live evaluation.",
        )

    # Validate spec exists
    specs_dir = settings.data_dir / "golden" / "specs"
    specs = load_all_specs(specs_dir)
    spec = next((s for s in specs if s.shot_id == req.spec_id), None)
    if spec is None:
        raise HTTPException(404, f"Spec not found: {req.spec_id}")

    run = await store.create_run(
        RunType.EVALUATION,
        config={"image_url": req.image_url, "spec_id": req.spec_id},
    )

    background_tasks.add_task(
        _run_evaluation_job, run.id, req.image_url, req.spec_id
    )

    return {"run_id": run.id}


@router.post("/evaluate/upload", status_code=202)
async def evaluate_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    spec_id: str = Form(...),
) -> dict[str, str]:
    """Start an evaluation from an uploaded image file."""
    settings = get_settings()
    if settings.replay_mode or not settings.live_ready:
        raise HTTPException(
            503, "Live evaluation unavailable — server is in replay mode",
        )
    thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
    if not thresholds_path.exists():
        raise HTTPException(
            503, "Calibration thresholds not found.",
        )

    # Validate content type
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            400, f"Invalid image type: {content_type}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            400,
            f"Image too large: {len(contents)} bytes (max {MAX_IMAGE_BYTES})",
        )

    # Validate spec
    specs_dir = settings.data_dir / "golden" / "specs"
    specs = load_all_specs(specs_dir)
    spec = next((s for s in specs if s.shot_id == spec_id), None)
    if spec is None:
        raise HTTPException(404, f"Spec not found: {spec_id}")

    # Save to temp file and upload to fal.ai
    suffix = Path(file.filename or "image.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    from product_fidelity_lab.storage.fal_storage import FalStorage

    fal_storage = FalStorage(
        cache_file=settings.data_dir / "cache" / "upload_cache.json",
    )
    image_url = await fal_storage.upload_image(tmp_path)
    tmp_path.unlink(missing_ok=True)

    store = get_run_store()
    run = await store.create_run(
        RunType.EVALUATION,
        config={"image_url": image_url, "spec_id": spec_id, "source": "upload"},
    )

    background_tasks.add_task(
        _run_evaluation_job, run.id, image_url, spec_id,
    )

    return {"run_id": run.id}


async def _run_evaluation_job(
    run_id: str,
    image_url: str,
    spec_id: str,
) -> None:
    """Background task for evaluation."""
    settings = get_settings()
    store = get_run_store()

    try:
        thresholds_path = settings.data_dir / "calibration" / "thresholds.json"
        grade_thresholds = load_thresholds(thresholds_path)
        specs_dir = settings.data_dir / "golden" / "specs"
        specs = load_all_specs(specs_dir)
        spec = next(s for s in specs if s.shot_id == spec_id)

        fal = FalClient(
            timeout_s=settings.fal_timeout_s,
            max_concurrent=settings.fal_max_concurrent,
        )

        # Pre-load golden image from disk if URL matches (avoids CDN download)
        if spec.image_url and image_url == spec.image_url:
            local_golden = settings.data_dir / "golden" / spec.image_path
            from product_fidelity_lab.evaluation.image_fetch import preload_local

            preload_local(image_url, local_golden)

        # Load pre-computed golden depth map (from prepare_golden.py)
        import numpy as np

        depth_cache = settings.data_dir / "cache" / "golden_depth" / f"{spec_id}.npy"
        if depth_cache.exists():
            golden_depth = np.load(str(depth_cache))
        else:
            golden_image_url = spec.image_url
            if golden_image_url is None:
                raise ValueError(f"Golden spec {spec_id} has no image_url")
            golden_depth = await get_depth_map(golden_image_url, fal)
            depth_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(depth_cache), golden_depth)

        await run_evaluation(
            image_url,
            spec,
            golden_depth,
            fal_client=fal,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            run_store=store,
            run_id=run_id,
            grade_thresholds=grade_thresholds,
            thresholds_source=str(thresholds_path),
            cache_root=settings.data_dir / "cache" / "results",
        )
    except Exception as exc:
        logger.error("evaluation.job_failed", run_id=run_id, error=str(exc))
        from product_fidelity_lab.models.run import RunStatus

        await store.update_status(run_id, RunStatus.FAILED)


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get run status and results."""
    store = get_run_store()
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run.model_dump()


@router.get("/runs")
async def list_runs(
    type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List runs, optionally filtered by type."""
    store = get_run_store()
    type_filter = RunType(type) if type else None
    runs = await store.list_runs(type_filter=type_filter, limit=limit)
    return [r.model_dump() for r in runs]
