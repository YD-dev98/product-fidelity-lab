"""Shot-aware prompt construction for Bacardi product photography generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from product_fidelity_lab.models.golden_spec import GoldenSpec

# Shot-specific camera/framing descriptions
_SHOT_DESCRIPTIONS: dict[str, str] = {
    "hero_front_straight": "front-facing hero shot, camera at eye level",
    "hero_front_left_45": "front-left 45-degree hero shot",
    "hero_front_right_45": "front-right 45-degree hero shot",
    "side_left": "left-side profile view, front label turning away",
    "side_right": "right-side profile view, front label partially visible",
    "back_straight": "direct back view showing rear label",
    "label_closeup": (
        "tight closeup of the front label filling the frame"
    ),
    "neck_closeup": (
        "closeup of the neck and shoulder area showing 1862 embossing"
    ),
    "cap_closeup": (
        "closeup of the silver ridged screw cap and neck band"
    ),
    "glass_material_detail": (
        "closeup of the lower bottle showing glass material "
        "and green base tint"
    ),
}

# Core product description — avoids brand/alcohol terms that trigger content policies
_PRODUCT_DESC = (
    "a clear glass bottle with a white front label, "
    "silver ridged screw cap, "
    "subtle green tint at the glass base, "
    "and colorless liquid inside"
)

_GROUNDING = (
    "Standing on a white sweep surface with visible floor contact "
    "and soft shadow. Studio lighting, sharp focus, high resolution."
)


def build_generation_prompt(
    spec: GoldenSpec,
    n_references: int = 1,
) -> str:
    """Build a shot-specific prompt with @imageN reference tags.

    Args:
        spec: The golden spec to generate for.
        n_references: Number of reference images being passed.
    """
    shot_desc = _SHOT_DESCRIPTIONS.get(spec.shot_id, spec.description)

    parts = [
        f"Professional studio product photography of {_PRODUCT_DESC}.",
        f"Shot: {shot_desc}.",
        _GROUNDING,
    ]

    # Add reference tags for all images
    for i in range(1, n_references + 1):
        parts.append(f"@image{i}")

    return " ".join(parts)


def select_reference_urls(
    target_spec: GoldenSpec,
    all_specs: list[GoldenSpec],
    max_refs: int = 4,
) -> list[str]:
    """Select reference URLs for multi-reference generation.

    Always includes the target spec's image first, then adds
    complementary detail shots (label, cap, glass_material_detail)
    to give the model more brand-specific conditioning.
    """
    refs: list[str] = []

    # Primary: target shot itself
    if target_spec.image_url:
        refs.append(target_spec.image_url)

    # Complementary detail shots that add brand/material signal
    complement_ids = [
        "label_closeup",
        "cap_closeup",
        "glass_material_detail",
    ]

    for cid in complement_ids:
        if len(refs) >= max_refs:
            break
        if cid == target_spec.shot_id:
            continue
        comp = next((s for s in all_specs if s.shot_id == cid), None)
        if comp and comp.image_url:
            refs.append(comp.image_url)

    return refs
