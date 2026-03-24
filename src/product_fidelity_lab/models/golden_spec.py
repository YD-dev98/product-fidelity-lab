"""Frozen golden specification for a product photography shot."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class AtomicFact(BaseModel):
    id: str
    category: Literal[
        "GEOMETRY", "MATERIAL", "LIGHTING", "COLOR", "DETAIL", "TEXT", "CONTEXT"
    ]
    fact: str
    critical: bool = False
    importance: Literal["high", "medium", "low"] = "medium"


class ROI(BaseModel):
    """Region of interest for product-aware cropping (normalized 0-1)."""

    x: float
    y: float
    width: float
    height: float
    label: str


class ExpectedText(BaseModel):
    text: str
    critical: bool = False
    match_mode: Literal["exact_token", "fuzzy"] = "exact_token"

    @model_validator(mode="after")
    def critical_requires_exact(self) -> ExpectedText:
        if self.critical and self.match_mode != "exact_token":
            msg = "critical=True requires match_mode='exact_token'"
            raise ValueError(msg)
        return self


class GoldenSpec(BaseModel):
    shot_id: str
    image_path: str
    image_url: str | None = None
    category: Literal["training", "validation"]
    shot_type: str

    atomic_facts: list[AtomicFact]
    expected_texts: list[ExpectedText]
    brand_colors_hex: list[str]
    rois: list[ROI]

    spec_version: int = 1

    description: str
    challenge_type: str | None = None
