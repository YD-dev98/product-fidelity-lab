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


@router.delete("/products/{product_id}")
async def delete_product(product_id: str) -> dict[str, bool]:
    store = get_product_store()
    product = await store.get_product(product_id)
    if product is None:
        raise HTTPException(404, "Product not found")
    await store.delete_product(product_id)
    return {"deleted": True}


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
