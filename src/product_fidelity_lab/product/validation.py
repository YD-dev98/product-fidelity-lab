"""Preset-product compatibility validation.

Extracted to avoid circular imports between api/ and main.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from product_fidelity_lab.models.preset import StudioPreset
from product_fidelity_lab.models.product import ProductAsset, ProductProfile
from product_fidelity_lab.storage.product_store import is_usable_reference

if TYPE_CHECKING:
    from product_fidelity_lab.storage.product_store import ProductStore


async def validate_compatibility(
    profile: ProductProfile,
    preset: StudioPreset,
    product_store: ProductStore | None,
    product_id: str,
) -> list[str]:
    """Validate preset-product compatibility.

    Returns a list of strings:
    - Prefixed with "ERROR:" for hard blockers (should reject the render).
    - Plain strings for warnings (render can proceed).
    """
    warnings: list[str] = []

    # Packaging type check
    if (
        preset.compatible_packaging_types is not None
        and profile.packaging_type not in preset.compatible_packaging_types
    ):
        warnings.append(
            f"ERROR: Preset '{preset.preset_id}' is not compatible with "
            f"packaging type '{profile.packaging_type.value}'"
        )

    # Required views check
    if preset.required_views:
        available_angles = set(profile.views.keys())
        missing = [v for v in preset.required_views if v not in available_angles]
        if missing:
            missing_str = ", ".join(v.value for v in missing)
            warnings.append(
                f"ERROR: Missing required views: {missing_str}"
            )

    # Min reference count — counts ALL usable refs regardless of angle.
    # required_views is enforced separately above via the profile.views mapping.
    if product_store is not None:
        assets: list[ProductAsset] = await product_store.get_assets(product_id)
        usable_count = sum(
            1 for a in assets if is_usable_reference(a)
        )
        if usable_count < preset.min_reference_count:
            warnings.append(
                f"ERROR: Need {preset.min_reference_count} usable reference(s), "
                f"found {usable_count}"
            )

    # Preferred views (warn only)
    if preset.preferred_views:
        available_angles = set(profile.views.keys())
        missing_pref = [v for v in preset.preferred_views if v not in available_angles]
        if missing_pref:
            missing_str = ", ".join(v.value for v in missing_pref)
            warnings.append(f"Missing preferred views: {missing_str}")

    # Ingest confidence
    if profile.ingest_confidence < 0.3:
        warnings.append(
            f"Low ingest confidence ({profile.ingest_confidence:.0%}) — "
            "consider uploading more images"
        )

    return warnings
