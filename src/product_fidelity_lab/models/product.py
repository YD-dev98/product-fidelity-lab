"""Product data models and enums."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProductStatus(StrEnum):
    DRAFT = "draft"
    PROFILED = "profiled"
    READY = "ready"


class AngleTag(StrEnum):
    FRONT = "front"
    FRONT_LEFT_45 = "front_left_45"
    FRONT_RIGHT_45 = "front_right_45"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    BACK = "back"
    DETAIL = "detail"
    TOP = "top"
    UNKNOWN = "unknown"


class PackagingType(StrEnum):
    BOTTLE = "bottle"
    BOX = "box"
    TUBE = "tube"
    JAR = "jar"
    CAN = "can"
    POUCH = "pouch"
    UNKNOWN = "unknown"


class AssetType(StrEnum):
    RAW_UPLOAD = "raw_upload"
    CLEANED = "cleaned"
    ALPHA_MASK = "alpha_mask"
    MASK = "mask"
    LOGO_CROP = "logo_crop"
    LABEL_CROP = "label_crop"


class ProviderMode(StrEnum):
    REFERENCE_ONLY = "reference_only"
    ADAPTER = "adapter"
    FINETUNE = "finetune"


class ModelStatus(StrEnum):
    NONE = "none"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"


class Product(BaseModel):
    """Lightweight product shell — profile and model live in related tables."""

    id: str
    name: str
    status: ProductStatus = ProductStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ProductAsset(BaseModel):
    id: str
    product_id: str
    asset_type: AssetType
    file_path: str
    fal_url: str | None = None
    angle_tag: AngleTag = AngleTag.UNKNOWN
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ProductProfile(BaseModel):
    """Canonical product identity. Stored in products.profile_json."""

    product_id: str
    brand_texts: list[str] = Field(default_factory=list)
    critical_texts: list[str] = Field(default_factory=list)
    brand_colors_hex: list[str] = Field(default_factory=list)
    logo_asset_id: str | None = None
    label_asset_id: str | None = None
    alpha_mask_asset_id: str | None = None
    views: dict[AngleTag, str] = Field(default_factory=dict)
    depth_summary: str | None = None
    material_summary: str | None = None
    packaging_type: PackagingType = PackagingType.UNKNOWN
    ingest_confidence: float = 0.0
    profile_version: int = 1


class ProductModel(BaseModel):
    """Tracks an external provider-backed model for this product."""

    product_id: str
    provider: str = "fal"
    mode: ProviderMode = ProviderMode.REFERENCE_ONLY
    external_model_id: str | None = None
    strength: float = 0.8
    status: ModelStatus = ModelStatus.NONE
    trained_on_n_images: int = 0
    trigger_word: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
