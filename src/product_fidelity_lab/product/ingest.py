"""Product image ingestion pipeline.

Accepts uploaded product images and enriches them with:
- Background removal + alpha mask (optional, non-blocking)
- OCR text extraction
- Dominant color extraction
- Batched angle tagging via Gemini
- Draft product profile
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from PIL import Image

from product_fidelity_lab.models.product import (
    AngleTag,
    AssetType,
    ProductAsset,
    ProductProfile,
    ProductStatus,
)
from product_fidelity_lab.models.run import RunStatus
from product_fidelity_lab.product.profile import ProductProfileBuilder

if TYPE_CHECKING:
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.storage.product_store import ProductStore
    from product_fidelity_lab.storage.run_store import RunStore

logger = structlog.get_logger()

REMBG_MODEL = "fal-ai/imageutils/rembg"

ANGLE_TAG_PROMPT = """You are analyzing product photography. For each numbered image below,
classify its camera angle as exactly one of these values:
front, front_left_45, front_right_45, side_left, side_right, back, detail, top, unknown

Consider the images relative to each other to distinguish angles consistently
(e.g. if two images show the front at slightly different rotations, pick the closest match).

Respond with a JSON array of objects, one per image, in the same order:
[
  {{"image": 1, "angle": "front", "reasoning": "..."}},
  ...
]

Only output the JSON array, no other text."""


async def ingest_product_images(
    product_id: str,
    file_paths: list[Path],
    *,
    product_store: ProductStore,
    fal_client: FalClient,
    gemini_api_key: str,
    gemini_model: str,
    run_store: RunStore,
    run_id: str,
) -> ProductProfile:
    """Full ingestion pipeline for uploaded product images.

    Always succeeds for raw uploads. Enrichments (bg removal, OCR, colors,
    angle tagging) are optional — failures are logged and reduce
    ingest_confidence but don't fail the pipeline.
    """
    await run_store.update_status(run_id, RunStatus.RUNNING)

    assets: list[ProductAsset] = []
    builder = ProductProfileBuilder(product_id, assets)

    # 1. Register raw uploads and get dimensions
    for file_path in file_paths:
        try:
            img = Image.open(file_path)
            w, h = img.size
        except Exception:
            logger.warning("ingest.bad_image", path=str(file_path))
            continue

        asset_id = uuid.uuid4().hex[:12]
        asset = ProductAsset(
            id=asset_id,
            product_id=product_id,
            asset_type=AssetType.RAW_UPLOAD,
            file_path=str(file_path),
            width=w,
            height=h,
        )
        await product_store.add_asset(asset)
        assets.append(asset)

    if not assets:
        await run_store.update_status(run_id, RunStatus.FAILED)
        return builder.build()

    # 2. Upload raw images to fal.ai for use in generation
    for asset in assets:
        try:
            import fal_client as fal_mod

            url: str = await fal_mod.upload_file_async(Path(asset.file_path))  # type: ignore[reportUnknownMemberType]
            await product_store.update_asset(asset.id, fal_url=url)
            asset.fal_url = url
        except Exception:
            logger.warning("ingest.upload_failed", asset_id=asset.id)

    # 3. Background removal + alpha mask (optional enrichment)
    alpha_mask_id: str | None = None
    for asset in list(assets):
        if not asset.fal_url:
            continue
        try:
            result = await fal_client.subscribe(
                REMBG_MODEL, {"image_url": asset.fal_url}
            )
            cleaned_url = str(result["image"]["url"])  # type: ignore[index]

            mask_id = uuid.uuid4().hex[:12]
            mask_asset = ProductAsset(
                id=mask_id,
                product_id=product_id,
                asset_type=AssetType.ALPHA_MASK,
                file_path=cleaned_url,  # stored as URL reference
                fal_url=cleaned_url,
                width=asset.width,
                height=asset.height,
                angle_tag=asset.angle_tag,
            )
            await product_store.add_asset(mask_asset)
            assets.append(mask_asset)
            if alpha_mask_id is None:
                alpha_mask_id = mask_id
            builder.set_alpha_mask(mask_id)
        except Exception:
            logger.warning("ingest.rembg_failed", asset_id=asset.id)

    # 4. OCR text extraction (reuse evaluation infrastructure)
    all_texts: list[str] = []
    raw_assets = [a for a in assets if a.asset_type == AssetType.RAW_UPLOAD]
    for asset in raw_assets:
        if not asset.fal_url:
            continue
        try:
            from product_fidelity_lab.evaluation.layer_brand import extract_text

            texts = await extract_text(asset.fal_url, fal_client)
            all_texts.extend(texts)
        except Exception:
            logger.warning("ingest.ocr_failed", asset_id=asset.id)
    builder.set_brand_texts(list(set(all_texts)))

    # 5. Dominant color extraction (reuse evaluation infrastructure)
    colors_hex: list[str] = []
    for asset in raw_assets[:1]:  # Use first image for palette
        try:
            import numpy as np
            from skimage.color import lab2rgb  # type: ignore[import-untyped]

            from product_fidelity_lab.evaluation.color import (
                extract_dominant_colors,
                rgb_to_hex,
            )

            img = Image.open(asset.file_path)
            lab_colors = extract_dominant_colors(img, n_colors=5)
            for lab in lab_colors:
                lab_3d = lab.reshape(1, 1, 3)
                rgb_arr = np.asarray(lab2rgb(lab_3d), dtype=np.float64).reshape(3) * 255.0
                colors_hex.append(rgb_to_hex(rgb_arr))
        except Exception:
            logger.warning("ingest.color_failed", asset_id=asset.id)
    builder.set_colors(colors_hex)

    # 6. Batched angle tagging via Gemini
    await _tag_angles_batched(
        raw_assets, gemini_api_key, gemini_model, product_store
    )
    # Refresh assets list after angle updates
    refreshed = await product_store.get_assets(product_id)
    assets.clear()
    assets.extend(refreshed)
    # Re-init builder with refreshed assets, re-applying all enrichments
    builder = ProductProfileBuilder(product_id, assets)
    builder.set_brand_texts(list(set(all_texts)))
    builder.set_colors(colors_hex)
    if alpha_mask_id is not None:
        builder.set_alpha_mask(alpha_mask_id)

    # 7. Build and store profile
    profile = builder.build()
    await product_store.update_product(
        product_id,
        status=ProductStatus.PROFILED,
        profile=profile,
    )

    await run_store.update_status(run_id, RunStatus.COMPLETE)
    await run_store.update_result(
        run_id,
        result={"product_id": product_id, "asset_count": len(assets)},
    )

    logger.info(
        "ingest.complete",
        product_id=product_id,
        asset_count=len(assets),
        confidence=profile.ingest_confidence,
    )
    return profile


async def _tag_angles_batched(
    assets: list[ProductAsset],
    gemini_api_key: str,
    gemini_model: str,
    product_store: ProductStore,
) -> None:
    """Classify all images in a single batched Gemini call."""
    if not assets:
        return

    urls_with_data: list[tuple[ProductAsset, bytes]] = []
    for asset in assets:
        try:
            img_bytes = Path(asset.file_path).read_bytes()
            urls_with_data.append((asset, img_bytes))
        except Exception:
            logger.warning("ingest.angle_read_failed", asset_id=asset.id)

    if not urls_with_data:
        return

    try:
        from google import genai

        client = genai.Client(api_key=gemini_api_key)

        contents: list[Any] = []
        for i, (asset, img_bytes) in enumerate(urls_with_data, 1):
            mime_type = _detect_mime(asset.file_path)
            contents.append(f"Image {i}:")
            contents.append(
                genai.types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            )
        contents.append(ANGLE_TAG_PROMPT)

        response = await client.aio.models.generate_content(
            model=gemini_model,
            contents=contents,
        )

        text = (response.text or "").strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        results: list[dict[str, Any]] = json.loads(text)

        valid_tags = {t.value for t in AngleTag}
        for i, item in enumerate(results):
            if i >= len(urls_with_data):
                break
            angle_str = str(item.get("angle", "unknown")).lower()
            if angle_str not in valid_tags:
                angle_str = "unknown"
            tag = AngleTag(angle_str)
            asset = urls_with_data[i][0]
            await product_store.update_asset(asset.id, angle_tag=tag)
            asset.angle_tag = tag

        logger.info("ingest.angles_tagged", count=len(results))

    except Exception:
        logger.warning("ingest.angle_tagging_failed", count=len(urls_with_data))


def _detect_mime(file_path_or_url: str) -> str:
    """Detect MIME type from file extension, stripping query strings for URLs."""
    from urllib.parse import urlparse

    # Strip query/fragment for signed URLs like foo.webp?sig=...
    path = urlparse(file_path_or_url).path if "://" in file_path_or_url else file_path_or_url
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
