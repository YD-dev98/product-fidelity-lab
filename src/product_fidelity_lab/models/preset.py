"""Studio preset, render request, and result models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from product_fidelity_lab.models.product import (
    AngleTag,
    PackagingType,
    ProductProfile,
)


class CameraAngle(StrEnum):
    FRONT = "front"
    FRONT_LEFT_45 = "front_left_45"
    FRONT_RIGHT_45 = "front_right_45"
    OVERHEAD_45 = "overhead_45"
    TOP_DOWN = "top_down"
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"


class CropMode(StrEnum):
    FULL_PRODUCT = "full_product"
    HERO_CENTER = "hero_center"
    CLOSE_LABEL = "close_label"
    LIFESTYLE_WIDE = "lifestyle_wide"


class PropPolicy(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    CONTEXTUAL = "contextual"


class RepairAction(StrEnum):
    LOGO_COMPOSITE = "logo_composite"
    LABEL_COMPOSITE = "label_composite"
    LOCALIZED_EDIT = "localized_edit"


class StudioPreset(BaseModel):
    """Controlled studio parameters replacing open-ended prompting."""

    preset_id: str
    name: str
    description: str = ""
    background: str = "white sweep"
    surface: str = "white surface"
    lighting: str = "soft studio, even, diffused"
    camera_angle: CameraAngle = CameraAngle.FRONT
    crop: CropMode = CropMode.HERO_CENTER
    aspect_ratio: str = "1:1"
    prop_policy: PropPolicy = PropPolicy.NONE
    style_modifiers: list[str] = Field(default_factory=list)
    guidance_scale: float | None = None
    num_inference_steps: int | None = None
    compatible_packaging_types: list[PackagingType] | None = None
    required_views: list[AngleTag] = Field(default_factory=list)
    preferred_views: list[AngleTag] = Field(default_factory=list)
    min_reference_count: int = 1
    supports_text_critical: bool = True


class RenderRequest(BaseModel):
    """Main user-facing request: product + preset -> result."""

    product_id: str
    preset_id: str
    style_prompt: str = ""
    aspect_ratio: str | None = None
    num_candidates: int = Field(default=4, ge=1, le=12)
    seed: int | None = None
    skip_repair: bool = False
    skip_ranking: bool = False


class Candidate(BaseModel):
    """One generation attempt within a render job."""

    candidate_id: str
    image_url: str
    seed: int
    model_id: str
    strategy_used: str
    generation_ms: int
    cost_estimate: float
    # Stage 1: sanity filter results
    filter_passed: bool | None = None
    filter_reasons: list[str] = Field(default_factory=list)
    # Stage 2: Gemini ranker scores
    rank_score: float | None = None
    identity_score: float | None = None
    label_score: float | None = None
    style_score: float | None = None
    composition_score: float | None = None


class RenderTrace(BaseModel):
    """Full reproducibility record for a render job."""

    final_prompt: str = ""
    selected_reference_asset_ids: list[str] = Field(default_factory=list)
    selected_reference_urls: list[str] = Field(default_factory=list)
    provider_payload: dict[str, Any] = Field(default_factory=dict)
    strategy_used: str = ""
    seeds: list[int] = Field(default_factory=list)
    repair_actions: list[RepairAction] = Field(default_factory=list)
    preset_snapshot: StudioPreset | None = None
    profile_snapshot: ProductProfile | None = None
    # Ranking trace
    ranker_version: str | None = None
    filter_summary: dict[str, Any] = Field(default_factory=dict)
    judge_summary: dict[str, Any] = Field(default_factory=dict)


class RenderResult(BaseModel):
    """Final output of a render job."""

    run_id: str
    product_id: str
    preset_id: str
    candidates: list[Candidate] = Field(default_factory=list)
    selected: list[Candidate] = Field(default_factory=list)
    repaired: list[Candidate] = Field(default_factory=list)
    final_image_url: str | None = None
    total_duration_ms: int = 0
    total_cost: float = 0.0
    trace: RenderTrace = Field(default_factory=RenderTrace)
    warnings: list[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    """Deterministic repair operations for a candidate."""

    candidate_id: str
    actions: list[RepairAction] = Field(default_factory=list)
    logo_asset_id: str | None = None
    label_asset_id: str | None = None
    edit_prompt: str | None = None
    edit_mask_path: str | None = None
