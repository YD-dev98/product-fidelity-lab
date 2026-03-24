"""Perturbation harness — apply controlled degradations and evaluate."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from product_fidelity_lab.models.golden_spec import ROI


class PerturbationType(StrEnum):
    BLUR = "blur"
    CROP_SHIFT = "crop_shift"
    HUE_ROTATION = "hue_rotation"
    TEXT_REMOVAL = "text_removal"
    BRIGHTNESS_REDUCTION = "brightness_reduction"


class PerturbationCase(BaseModel):
    perturbation_type: str
    expected_failures: list[str]
    actual_failures: list[str] = []
    caught: bool = False
    score: float | None = None
    grade: str | None = None


class PerturbationReport(BaseModel):
    cases: list[PerturbationCase]
    detection_rate: float = 0.0


DEFAULT_PERTURBATIONS: list[dict[str, Any]] = [
    {
        "type": PerturbationType.BLUR,
        "params": {"sigma": 3},
        "expected_failures": ["DETAIL", "MATERIAL"],
    },
    {
        "type": PerturbationType.CROP_SHIFT,
        "params": {"offset_pct": 0.10},
        "expected_failures": ["GEOMETRY"],
    },
    {
        "type": PerturbationType.HUE_ROTATION,
        "params": {"degrees": 30},
        "expected_failures": ["COLOR"],
    },
    {
        "type": PerturbationType.TEXT_REMOVAL,
        "params": {},
        "expected_failures": ["TEXT", "brand_text"],
    },
    {
        "type": PerturbationType.BRIGHTNESS_REDUCTION,
        "params": {"factor": 0.4},
        "expected_failures": ["LIGHTING"],
    },
]


def apply_perturbation(
    image_path: Path,
    perturbation_type: PerturbationType,
    params: dict[str, Any] | None = None,
    rois: list[ROI] | None = None,
) -> Image.Image:
    """Apply a single perturbation to an image."""
    img = Image.open(image_path).convert("RGB")
    params = params or {}

    if perturbation_type == PerturbationType.BLUR:
        sigma = params.get("sigma", 3)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    if perturbation_type == PerturbationType.CROP_SHIFT:
        offset = params.get("offset_pct", 0.10)
        w, h = img.size
        dx = int(w * offset)
        dy = int(h * offset)
        # Shift and pad with black
        shifted = Image.new("RGB", (w, h), (0, 0, 0))
        shifted.paste(img, (dx, dy))
        return shifted

    if perturbation_type == PerturbationType.HUE_ROTATION:
        degrees = params.get("degrees", 30)
        arr = np.array(img)
        hsv = Image.fromarray(arr).convert("HSV")
        hsv_arr = np.array(hsv)
        hsv_arr[:, :, 0] = (hsv_arr[:, :, 0].astype(int) + int(degrees * 255 / 360)) % 256
        rotated = Image.fromarray(hsv_arr, "HSV").convert("RGB")
        return rotated

    if perturbation_type == PerturbationType.TEXT_REMOVAL:
        # Fill label ROI with average surrounding color
        label_roi = None
        if rois:
            label_roi = next((r for r in rois if r.label in ("label", "logo")), None)
        if label_roi is None:
            # Default: fill center 30% region
            w, h = img.size
            from product_fidelity_lab.models.golden_spec import ROI

            label_roi = ROI(x=0.35, y=0.35, width=0.30, height=0.30, label="label")
        arr = np.array(img)
        ih, iw = arr.shape[:2]
        x1 = int(label_roi.x * iw)
        y1 = int(label_roi.y * ih)
        x2 = int((label_roi.x + label_roi.width) * iw)
        y2 = int((label_roi.y + label_roi.height) * ih)
        # Average color around the ROI border
        border = np.concatenate([
            arr[max(0, y1 - 5) : y1, x1:x2].reshape(-1, 3),
            arr[y2 : min(ih, y2 + 5), x1:x2].reshape(-1, 3),
        ])
        if len(border) > 0:
            avg_color = border.mean(axis=0).astype(np.uint8)
        else:
            avg_color = np.array([128, 128, 128], dtype=np.uint8)
        arr[y1:y2, x1:x2] = avg_color
        return Image.fromarray(arr)

    if perturbation_type == PerturbationType.BRIGHTNESS_REDUCTION:
        factor = params.get("factor", 0.4)
        arr = np.array(img, dtype=np.float64)
        arr = (arr * factor).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    msg = f"Unknown perturbation: {perturbation_type}"
    raise ValueError(msg)


def save_perturbed(img: Image.Image, output_path: Path) -> Path:
    """Save a perturbed image and return the path."""
    img.save(output_path, quality=90)
    return output_path
