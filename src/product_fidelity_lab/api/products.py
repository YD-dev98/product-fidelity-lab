"""Product management API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.main import get_product_store, get_run_store
from product_fidelity_lab.models.product import AngleTag, PackagingType
from product_fidelity_lab.models.run import RunStatus, RunType

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["products"])

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


class CreateProductRequest(BaseModel):
    name: str


class ConfirmProfileRequest(BaseModel):
    critical_texts: list[str] | None = None
    packaging_type: PackagingType | None = None
    angle_overrides: dict[str, AngleTag] | None = None


@router.post("/products", status_code=201)
async def create_product(req: CreateProductRequest) -> dict[str, str]:
    store = get_product_store()
    product = await store.create_product(req.name)
    return {"product_id": product.id}


@router.get("/products")
async def list_products(limit: int = 50) -> list[dict[str, Any]]:
    store = get_product_store()
    products = await store.list_products(limit=limit)
    return [p.model_dump() for p in products]


@router.get("/products/{product_id}")
async def get_product(product_id: str) -> dict[str, Any]:
    store = get_product_store()
    product = await store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    profile = await store.get_profile(product_id)
    model = await store.get_model(product_id)
    result = product.model_dump()
    result["profile"] = profile.model_dump() if profile else None
    result["model"] = model.model_dump() if model else None
    return result


@router.post("/products/{product_id}/upload")
async def upload_images(
    product_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),  # noqa: B008
) -> Any:
    """Upload product images. Returns 202 if at least one accepted, 422 if none."""
    settings = get_settings()
    product_store = get_product_store()

    product = await product_store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")

    # Count check
    if len(files) > settings.max_upload_images:
        raise HTTPException(
            400,
            f"Too many files: {len(files)} (max {settings.max_upload_images})",
        )

    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    saved_paths: list[Path] = []

    uploads_dir = product_store.uploads_dir(product_id)

    for f in files:
        filename = f.filename or "unknown"

        # MIME check
        content_type = (f.content_type or "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_MIME_TYPES:
            rejected.append({"filename": filename, "reason": f"Invalid type: {content_type}"})
            continue

        # Read + size check
        contents = await f.read()
        if len(contents) > MAX_FILE_BYTES:
            rejected.append({"filename": filename, "reason": f"Too large: {len(contents)} bytes"})
            continue

        # Validate it's actually an image
        try:
            from io import BytesIO

            from PIL import Image
            Image.open(BytesIO(contents)).verify()
        except Exception:
            rejected.append({"filename": filename, "reason": "Not a valid image"})
            continue

        # Save to disk
        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(
            dir=str(uploads_dir), suffix=suffix, delete=False
        ) as tmp:
            tmp.write(contents)
            saved_path = Path(tmp.name)

        saved_paths.append(saved_path)
        accepted.append(filename)

    # Zero accepted -> 422
    if not saved_paths:
        raise HTTPException(422, {"accepted": [], "rejected": rejected})

    # Start background ingest
    if not settings.live_ready:
        raise HTTPException(
            503, "Live mode unavailable — API keys not configured"
        )

    run_store = get_run_store()
    run = await run_store.create_run(
        RunType.INGEST,
        config={"product_id": product_id, "file_count": len(saved_paths)},
    )

    background_tasks.add_task(
        _run_ingest_job, product_id, saved_paths, run.id
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=202,
        content={
            "run_id": run.id,
            "accepted": accepted,
            "rejected": rejected,
        },
    )


@router.get("/products/{product_id}/assets")
async def list_assets(
    product_id: str,
    type: str | None = None,
) -> list[dict[str, Any]]:
    store = get_product_store()
    from product_fidelity_lab.models.product import AssetType

    asset_type = AssetType(type) if type else None
    assets = await store.get_assets(product_id, asset_type=asset_type)
    return [a.model_dump() for a in assets]


@router.get("/products/{product_id}/profile")
async def get_profile(product_id: str) -> dict[str, Any]:
    store = get_product_store()
    profile = await store.get_profile(product_id)
    if profile is None:
        raise HTTPException(404, "Profile not found — upload images first")
    return profile.model_dump()


@router.post("/products/{product_id}/profile/confirm")
async def confirm_profile(
    product_id: str,
    req: ConfirmProfileRequest,
) -> dict[str, Any]:
    """Confirm/override profile fields after ingest."""
    store = get_product_store()
    profile = await store.get_profile(product_id)
    if profile is None:
        raise HTTPException(404, "Profile not found — upload images first")

    # Apply overrides
    if req.critical_texts is not None:
        profile.critical_texts = req.critical_texts
    if req.packaging_type is not None:
        profile.packaging_type = req.packaging_type

    # Apply angle overrides
    if req.angle_overrides:
        for asset_id, angle_tag in req.angle_overrides.items():
            # Verify asset belongs to this product
            asset = await store.get_asset(asset_id)
            if asset is None or asset.product_id != product_id:
                raise HTTPException(
                    422, f"Asset '{asset_id}' not found or does not belong to this product"
                )

            await store.update_asset(asset_id, angle_tag=angle_tag)

            # Remove any existing views entry pointing to this asset (avoids stale mappings)
            profile.views = {
                k: v for k, v in profile.views.items() if v != asset_id
            }
            # Add new views entry if concrete angle
            if angle_tag != AngleTag.UNKNOWN:
                profile.views[angle_tag] = asset_id

    await store.update_product(product_id, profile=profile)
    return profile.model_dump()


class TrainRequest(BaseModel):
    trigger_word: str = "PRODSHOT"
    provider: str = "fal"


@router.post("/products/{product_id}/train", status_code=202)
async def train_model(
    product_id: str,
    req: TrainRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Start training a product model. Returns run_id."""
    settings = get_settings()
    if not settings.live_ready:
        raise HTTPException(503, "Live mode unavailable — API keys not configured")

    product_store = get_product_store()
    run_store = get_run_store()

    product = await product_store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")

    profile = await product_store.get_profile(product_id)
    if profile is None:
        raise HTTPException(422, "Product has no profile — upload images first")

    # Gather training image URLs
    from product_fidelity_lab.models.product import AssetType

    assets = await product_store.get_assets(product_id)
    image_urls = [
        a.fal_url for a in assets
        if a.asset_type in (AssetType.RAW_UPLOAD, AssetType.CLEANED)
        and a.fal_url
    ]
    if not image_urls:
        raise HTTPException(422, "No uploaded images with URLs available for training")

    run = await run_store.create_run(
        RunType.TRAIN,
        config={
            "product_id": product_id,
            "provider": req.provider,
            "trigger_word": req.trigger_word,
            "image_count": len(image_urls),
        },
    )

    background_tasks.add_task(
        _run_train_job, product_id, image_urls, req.trigger_word, req.provider, run.id
    )

    return {"run_id": run.id}


class EvaluateRenderRequest(BaseModel):
    render_run_id: str


@router.post("/products/{product_id}/evaluate", status_code=202)
async def evaluate_render(
    product_id: str,
    req: EvaluateRenderRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Run deep QA evaluation on a completed render.

    Builds a synthetic spec from the product profile and runs AFV + brand
    checks (no depth). Opt-in — separate from the lightweight QA in the
    render pipeline.
    """
    settings = get_settings()
    if not settings.live_ready:
        raise HTTPException(503, "Live mode unavailable — API keys not configured")

    product_store = get_product_store()
    run_store = get_run_store()

    product = await product_store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")

    profile = await product_store.get_profile(product_id)
    if profile is None:
        raise HTTPException(422, "Product has no profile")

    # Load the render run and verify it belongs to this product
    render_run = await run_store.get_run(req.render_run_id)
    if render_run is None:
        raise HTTPException(404, f"Render run not found: {req.render_run_id}")
    if render_run.type != RunType.RENDER:
        raise HTTPException(422, "Not a render run")
    if render_run.config.get("product_id") != product_id:
        raise HTTPException(422, "Render run does not belong to this product")
    if render_run.result is None:
        raise HTTPException(422, "Render run has no result — still running?")

    final_url = render_run.result.get("final_image_url")
    if not final_url:
        raise HTTPException(422, "Render run has no final image")

    eval_run = await run_store.create_run(
        RunType.EVALUATION,
        config={
            "product_id": product_id,
            "render_run_id": req.render_run_id,
            "image_url": final_url,
            "source": "product_evaluate",
        },
    )

    background_tasks.add_task(
        _run_product_eval_job, product_id, final_url, eval_run.id,
    )

    return {"run_id": eval_run.id}


@router.get("/products/{product_id}/model")
async def get_model(product_id: str) -> dict[str, Any]:
    store = get_product_store()
    model = await store.get_model(product_id)
    if model is None:
        raise HTTPException(404, "No model trained for this product")
    return model.model_dump()


async def _run_product_eval_job(
    product_id: str,
    image_url: str,
    run_id: str,
) -> None:
    """Background task for product evaluation."""
    settings = get_settings()
    product_store = get_product_store()
    run_store = get_run_store()

    try:
        await run_store.update_status(run_id, RunStatus.RUNNING)

        profile = await product_store.get_profile(product_id)
        if profile is None:
            raise ValueError("Profile not found")

        from product_fidelity_lab.evaluation.product_eval import (
            run_product_evaluation,
        )
        from product_fidelity_lab.generation.client import FalClient

        fal = FalClient(
            timeout_s=settings.fal_timeout_s,
            max_concurrent=settings.fal_max_concurrent,
        )

        result = await run_product_evaluation(
            image_url,
            profile,
            fal_client=fal,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
        )

        await run_store.update_status(run_id, RunStatus.COMPLETE)
        await run_store.update_result(
            run_id,
            result=result,
            score=result["overall"],
            grade=result["grade"],
            passed=result["passed"],
            duration_ms=result["duration_ms"],
        )

    except Exception as exc:
        logger.error("product_eval.job_failed", run_id=run_id, error=str(exc))
        await run_store.update_status(run_id, RunStatus.FAILED)


@router.delete("/products/{product_id}")
async def delete_product(product_id: str) -> dict[str, bool]:
    store = get_product_store()
    product = await store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    await store.delete_product(product_id)
    return {"deleted": True}


async def _run_train_job(
    product_id: str,
    image_urls: list[str],
    trigger_word: str,
    provider_name: str,
    run_id: str,
) -> None:
    """Background task for model training."""
    product_store = get_product_store()
    run_store = get_run_store()

    try:
        await run_store.update_status(run_id, RunStatus.RUNNING)

        from product_fidelity_lab.generation.client import FalClient
        from product_fidelity_lab.product.provider import get_provider

        fal = FalClient(
            timeout_s=600,  # Training can take several minutes
            max_concurrent=1,
        )
        provider = get_provider(provider_name, fal)

        model = await provider.start_training(
            product_id, image_urls, trigger_word, product_store,
        )

        from product_fidelity_lab.models.product import ModelStatus, ProductStatus

        if model.status == ModelStatus.READY:
            await run_store.update_status(run_id, RunStatus.COMPLETE)
            await run_store.update_result(
                run_id,
                result={
                    "product_id": product_id,
                    "model_id": model.external_model_id,
                    "status": model.status.value,
                },
            )
            # Update product status to ready
            await product_store.update_product(product_id, status=ProductStatus.READY)
        else:
            await run_store.update_status(run_id, RunStatus.FAILED)
            await run_store.update_result(
                run_id,
                result={"product_id": product_id, "status": model.status.value},
            )

    except Exception as exc:
        logger.error("train.job_failed", run_id=run_id, error=str(exc))
        await run_store.update_status(run_id, RunStatus.FAILED)


async def _run_ingest_job(
    product_id: str,
    file_paths: list[Path],
    run_id: str,
) -> None:
    """Background task for product ingestion."""
    settings = get_settings()
    product_store = get_product_store()
    run_store = get_run_store()

    try:
        from product_fidelity_lab.generation.client import FalClient
        from product_fidelity_lab.product.ingest import ingest_product_images

        fal = FalClient(
            timeout_s=settings.fal_timeout_s,
            max_concurrent=settings.fal_max_concurrent,
        )

        await ingest_product_images(
            product_id,
            file_paths,
            product_store=product_store,
            fal_client=fal,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            run_store=run_store,
            run_id=run_id,
        )
    except Exception as exc:
        logger.error("ingest.job_failed", run_id=run_id, error=str(exc))
        await run_store.update_status(run_id, RunStatus.FAILED)
