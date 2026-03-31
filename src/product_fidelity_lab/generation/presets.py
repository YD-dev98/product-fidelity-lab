"""Builtin studio presets and prompt compilation."""

from __future__ import annotations

from product_fidelity_lab.models.preset import (
    CameraAngle,
    CropMode,
    PropPolicy,
    StudioPreset,
)
from product_fidelity_lab.models.product import AngleTag, PackagingType, ProductProfile

BUILTIN_PRESETS: dict[str, StudioPreset] = {
    "clean_packshot": StudioPreset(
        preset_id="clean_packshot",
        name="Clean Packshot",
        description="White background, centered, e-commerce ready",
        background="pure white sweep",
        surface="white surface with soft shadow",
        lighting="soft studio, even, diffused, no harsh highlights",
        camera_angle=CameraAngle.FRONT,
        crop=CropMode.HERO_CENTER,
        aspect_ratio="1:1",
        prop_policy=PropPolicy.NONE,
        guidance_scale=3.5,
        required_views=[AngleTag.FRONT],
        preferred_views=[AngleTag.FRONT, AngleTag.DETAIL],
        min_reference_count=1,
        supports_text_critical=True,
    ),
    "soft_luxury_tabletop": StudioPreset(
        preset_id="soft_luxury_tabletop",
        name="Soft Luxury Tabletop",
        description="Warm marble/stone surface, soft side lighting",
        background="soft gradient, warm tones",
        surface="white marble surface",
        lighting="soft warm side light, gentle rim light, shallow depth of field",
        camera_angle=CameraAngle.FRONT_LEFT_45,
        crop=CropMode.HERO_CENTER,
        aspect_ratio="4:5",
        prop_policy=PropPolicy.MINIMAL,
        guidance_scale=4.0,
        compatible_packaging_types=[
            PackagingType.BOTTLE, PackagingType.BOX, PackagingType.TUBE,
        ],
        required_views=[AngleTag.FRONT],
        preferred_views=[AngleTag.FRONT, AngleTag.FRONT_LEFT_45],
        min_reference_count=2,
        supports_text_critical=False,
    ),
    "editorial_lifestyle": StudioPreset(
        preset_id="editorial_lifestyle",
        name="Editorial Lifestyle",
        description="Contextual setting, editorial photography style",
        background="blurred lifestyle setting",
        surface="natural surface",
        lighting="natural window light, cinematic",
        camera_angle=CameraAngle.FRONT_LEFT_45,
        crop=CropMode.LIFESTYLE_WIDE,
        aspect_ratio="16:9",
        prop_policy=PropPolicy.CONTEXTUAL,
        guidance_scale=4.5,
        required_views=[],
        preferred_views=[AngleTag.FRONT],
        min_reference_count=1,
        supports_text_critical=False,
    ),
    "detail_closeup": StudioPreset(
        preset_id="detail_closeup",
        name="Detail Closeup",
        description="Tight crop on label/logo area",
        background="white sweep",
        surface="white surface",
        lighting="ring light, even, sharp focus on label text",
        camera_angle=CameraAngle.FRONT,
        crop=CropMode.CLOSE_LABEL,
        aspect_ratio="1:1",
        prop_policy=PropPolicy.NONE,
        guidance_scale=3.5,
        required_views=[AngleTag.FRONT],
        preferred_views=[AngleTag.FRONT, AngleTag.DETAIL],
        min_reference_count=1,
        supports_text_critical=True,
    ),
    "ecommerce_white": StudioPreset(
        preset_id="ecommerce_white",
        name="E-commerce White",
        description="Pure white background, flat lighting, marketplace ready",
        background="pure white, seamless",
        surface="none visible",
        lighting="flat, even, shadowless",
        camera_angle=CameraAngle.FRONT,
        crop=CropMode.HERO_CENTER,
        aspect_ratio="1:1",
        prop_policy=PropPolicy.NONE,
        guidance_scale=3.5,
        required_views=[AngleTag.FRONT],
        preferred_views=[AngleTag.FRONT],
        min_reference_count=1,
        supports_text_critical=True,
    ),
}


def compile_prompt(
    preset: StudioPreset,
    profile: ProductProfile,
    n_references: int,
    style_prompt: str = "",
) -> str:
    """Compile a StudioPreset + ProductProfile into a generation prompt.

    Builds a structured prompt from preset fields and profile data,
    appending @imageN tags for references.
    """
    parts: list[str] = []

    # Product description
    if profile.packaging_type.value != "unknown":
        parts.append(f"A {profile.packaging_type.value}")
    else:
        parts.append("A product")

    if profile.material_summary:
        parts.append(f"({profile.material_summary})")

    # Scene setup from preset
    parts.append(f"on {preset.surface}")
    parts.append(f"with {preset.background} background.")
    parts.append(f"Lighting: {preset.lighting}.")
    parts.append(f"Camera angle: {preset.camera_angle.value.replace('_', ' ')}.")

    if preset.crop == CropMode.CLOSE_LABEL:
        parts.append("Tight crop on the product label/logo area.")
    elif preset.crop == CropMode.LIFESTYLE_WIDE:
        parts.append("Wide composition with environmental context.")

    if preset.prop_policy == PropPolicy.CONTEXTUAL:
        parts.append("Include contextual props that complement the product.")
    elif preset.prop_policy == PropPolicy.MINIMAL:
        parts.append("Minimal styling, subtle shadows only.")

    # Style modifiers
    if preset.style_modifiers:
        parts.append(", ".join(preset.style_modifiers) + ".")

    # User style prompt
    if style_prompt:
        parts.append(style_prompt)

    # Brand text hints
    if profile.critical_texts:
        texts = ", ".join(f'"{t}"' for t in profile.critical_texts)
        parts.append(f"The product label reads: {texts}.")

    # Reference tags
    for i in range(1, n_references + 1):
        parts.append(f"@image{i}")

    return " ".join(parts)
