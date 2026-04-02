"""Deterministic label/logo compositing for brand-critical areas.

Phase 3 scope: front-facing presets only, no perspective transforms.
Uses Pillow for overlay and edit_image() for localized fixes.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import structlog
from PIL import Image

from product_fidelity_lab.models.preset import (
    CameraAngle,
    Candidate,
    RepairAction,
    RepairPlan,
    StudioPreset,
)
from product_fidelity_lab.models.product import ProductProfile

if TYPE_CHECKING:
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.storage.product_store import ProductStore

logger = structlog.get_logger()

# Presets where front-facing compositing is safe
COMPOSITABLE_ANGLES = {
    CameraAngle.FRONT,
    CameraAngle.EYE_LEVEL,
}


def is_repair_eligible(preset: StudioPreset) -> bool:
    """Check if a preset supports repair (front-facing + text-critical)."""
    return (
        preset.supports_text_critical
        and preset.camera_angle in COMPOSITABLE_ANGLES
    )


async def plan_repair(
    candidate: Candidate,
    profile: ProductProfile,
    preset: StudioPreset,
) -> RepairPlan:
    """Analyze a candidate and build a repair plan.

    Decides which repair actions are needed based on profile assets
    and candidate filter/rank results.
    """
    actions: list[RepairAction] = []

    # Logo composite if profile has a logo crop and candidate had label issues
    if profile.logo_asset_id and _needs_logo_repair(candidate):
        actions.append(RepairAction.LOGO_COMPOSITE)

    # Label composite if profile has a label crop
    if profile.label_asset_id and _needs_label_repair(candidate):
        actions.append(RepairAction.LABEL_COMPOSITE)

    # Localized edit if there are style/composition issues but identity is ok
    if _needs_localized_edit(candidate):
        actions.append(RepairAction.LOCALIZED_EDIT)

    return RepairPlan(
        candidate_id=candidate.candidate_id,
        actions=actions,
        logo_asset_id=profile.logo_asset_id if RepairAction.LOGO_COMPOSITE in actions else None,
        label_asset_id=profile.label_asset_id if RepairAction.LABEL_COMPOSITE in actions else None,
        edit_prompt=(
            _build_edit_prompt(candidate, preset)
            if RepairAction.LOCALIZED_EDIT in actions
            else None
        ),
    )


async def execute_repair(
    candidate: Candidate,
    plan: RepairPlan,
    product_store: ProductStore,
    fal_client: FalClient,
) -> Candidate:
    """Execute a repair plan and return an updated candidate.

    Applies logo/label compositing via Pillow, then localized edit via
    the FLUX edit endpoint if needed.
    """
    if not plan.actions:
        return candidate


    current_url = candidate.image_url

    # Step 1: Logo/label compositing via Pillow
    if RepairAction.LOGO_COMPOSITE in plan.actions or RepairAction.LABEL_COMPOSITE in plan.actions:
        try:
            current_url = await _composite_overlay(
                base_url=current_url,
                plan=plan,
                product_store=product_store,
            )
        except Exception:
            logger.warning(
                "repair.composite_failed",
                candidate_id=candidate.candidate_id,
            )

    # Step 2: Localized edit via FLUX edit
    if RepairAction.LOCALIZED_EDIT in plan.actions and plan.edit_prompt:
        try:
            current_url = await _localized_edit(
                base_url=current_url,
                edit_prompt=plan.edit_prompt,
                fal_client=fal_client,
            )
        except Exception:
            logger.warning(
                "repair.edit_failed",
                candidate_id=candidate.candidate_id,
            )

    # Return updated candidate with new URL
    return Candidate(
        candidate_id=candidate.candidate_id,
        image_url=current_url,
        seed=candidate.seed,
        model_id=candidate.model_id,
        strategy_used=candidate.strategy_used,
        generation_ms=candidate.generation_ms,
        cost_estimate=candidate.cost_estimate,
        filter_passed=candidate.filter_passed,
        filter_reasons=candidate.filter_reasons,
        rank_score=candidate.rank_score,
        identity_score=candidate.identity_score,
        label_score=candidate.label_score,
        style_score=candidate.style_score,
        composition_score=candidate.composition_score,
    )


async def _composite_overlay(
    base_url: str,
    plan: RepairPlan,
    product_store: ProductStore,
) -> str:
    """Overlay logo/label onto the base image using Pillow.

    Simple center-bottom placement for front-facing shots.
    Returns a fal.ai URL for the composited image.
    """
    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    base_bytes = await fetch_image_bytes(base_url)
    base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
    bw, bh = base_img.size

    # Overlay each asset, using stored bbox if available
    for asset_id in [plan.logo_asset_id, plan.label_asset_id]:
        if asset_id is None:
            continue
        asset = await product_store.get_asset(asset_id)
        if asset is None or not asset.fal_url:
            continue

        overlay_bytes = await fetch_image_bytes(asset.fal_url)
        overlay = Image.open(BytesIO(overlay_bytes)).convert("RGBA")

        # Use stored bbox from ingest for placement, fall back to center
        bbox = asset.metadata.get("bbox")
        if bbox:
            # Place overlay at the original detected region
            target_w = int(bbox["width"] * bw)
            target_h = int(bbox["height"] * bh)
            if target_w > 0 and target_h > 0:
                overlay = overlay.resize(
                    (target_w, target_h), Image.LANCZOS,
                )
            x = int(bbox["x"] * bw)
            y = int(bbox["y"] * bh)
        else:
            # Fallback: scale to 40% width, center-bottom
            max_w = int(bw * 0.4)
            ow, oh = overlay.size
            if ow > max_w:
                ratio = max_w / ow
                overlay = overlay.resize(
                    (max_w, int(oh * ratio)), Image.LANCZOS,
                )
            ow, oh = overlay.size
            x = (bw - ow) // 2
            y = int(bh * 0.55)

        base_img.paste(overlay, (x, y), overlay)

    # Convert back to RGB and upload
    result_img = base_img.convert("RGB")
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    import tempfile
    from pathlib import Path

    import fal_client

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(buf.getvalue())
        tmp_path = Path(tmp.name)

    url: str = await fal_client.upload_file_async(tmp_path)
    tmp_path.unlink(missing_ok=True)

    logger.info("repair.composite_complete", base_url=base_url[:60])
    return url


async def _localized_edit(
    base_url: str,
    edit_prompt: str,
    fal_client: FalClient,
) -> str:
    """Run a localized edit via the FLUX edit endpoint."""
    from product_fidelity_lab.generation.edit import edit_image
    from product_fidelity_lab.models.generation import EditRequest

    request = EditRequest(
        base_image_url=base_url,
        reference_urls=[],
        prompt=edit_prompt,
    )
    result = await edit_image(request, fal_client)
    logger.info("repair.edit_complete", base_url=base_url[:60])
    return result.image_url


def _needs_logo_repair(candidate: Candidate) -> bool:
    """Heuristic: logo repair if label score is low."""
    if candidate.label_score is not None and candidate.label_score < 0.5:
        return True
    return "critical_text_absent" in candidate.filter_reasons


def _needs_label_repair(candidate: Candidate) -> bool:
    """Heuristic: label repair if label score is very low."""
    if candidate.label_score is not None and candidate.label_score < 0.4:
        return True
    return "critical_text_absent" in candidate.filter_reasons


def _needs_localized_edit(candidate: Candidate) -> bool:
    """Heuristic: localized edit if style/composition needs help but identity is fine."""
    if candidate.identity_score is not None and candidate.identity_score >= 0.7:
        if candidate.style_score is not None and candidate.style_score < 0.5:
            return True
        if candidate.composition_score is not None and candidate.composition_score < 0.5:
            return True
    return False


def _build_edit_prompt(candidate: Candidate, preset: StudioPreset) -> str:
    """Build a targeted edit prompt for localized fixes."""
    parts = ["Fix the lighting, style, and composition of this product photo."]
    parts.append(f"Target style: {preset.lighting}, {preset.background}.")
    if candidate.composition_score is not None and candidate.composition_score < 0.5:
        parts.append("Improve the framing and crop.")
    return " ".join(parts)
