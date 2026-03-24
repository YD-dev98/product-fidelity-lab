"""Shot-aware prompt construction for product photography generation."""

from __future__ import annotations

from collections.abc import Callable
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


def _image_tags(n: int) -> str:
    return " ".join(f"@image{i}" for i in range(1, n + 1))


def _prompt_label_emphasis(spec: GoldenSpec, n_references: int = 1) -> str:
    shot_desc = _SHOT_DESCRIPTIONS.get(spec.shot_id, spec.description)
    parts = [
        f"Professional studio product photography of {_PRODUCT_DESC}.",
        f"Shot: {shot_desc}.",
        "The front label text must be clearly visible and fully readable.",
        "Reproduce all printed text on the label exactly as shown in the reference images.",
        _GROUNDING,
        _image_tags(n_references),
    ]
    return " ".join(parts)


def _prompt_detailed_label(spec: GoldenSpec, n_references: int = 1) -> str:
    shot_desc = _SHOT_DESCRIPTIONS.get(spec.shot_id, spec.description)
    parts = [
        f"Professional studio product photography of {_PRODUCT_DESC}.",
        (
            "The bottle has a white rectangular front label with dark printed text, "
            "a small logo near the top of the label, and fine-print details below. "
            "All text on the label must be sharp, legible, and match the reference exactly."
        ),
        f"Shot: {shot_desc}.",
        _GROUNDING,
        _image_tags(n_references),
    ]
    return " ".join(parts)


def _prompt_reference_focused(spec: GoldenSpec, n_references: int = 1) -> str:
    shot_desc = _SHOT_DESCRIPTIONS.get(spec.shot_id, spec.description)
    parts = [
        f"Product photograph matching the reference images exactly. {shot_desc}.",
        _image_tags(n_references),
    ]
    return " ".join(parts)


PROMPT_STRATEGIES: dict[str, Callable[..., str]] = {
    "baseline": build_generation_prompt,
    "label_emphasis": _prompt_label_emphasis,
    "detailed_label": _prompt_detailed_label,
    "reference_focused": _prompt_reference_focused,
}

PROMPT_STRATEGY_NAMES: frozenset[str] = frozenset(PROMPT_STRATEGIES)


def build_prompt_with_strategy(
    spec: GoldenSpec,
    n_references: int = 1,
    strategy: str = "baseline",
) -> str:
    fn = PROMPT_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown prompt strategy {strategy!r}, "
            f"choose from {sorted(PROMPT_STRATEGY_NAMES)}"
        )
    return fn(spec, n_references)


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


def _dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _spec_url(shot_id: str, all_specs: list[GoldenSpec]) -> str | None:
    s = next((s for s in all_specs if s.shot_id == shot_id), None)
    return s.image_url if s else None


def _refs_label_heavy(
    target_spec: GoldenSpec, all_specs: list[GoldenSpec], max_refs: int = 4
) -> list[str]:
    urls: list[str] = []
    if target_spec.image_url:
        urls.append(target_spec.image_url)
    label_sids = [
        "label_closeup", "hero_front_straight",
        "hero_front_left_45", "hero_front_right_45",
    ]
    for sid in label_sids:
        if sid == target_spec.shot_id:
            continue
        u = _spec_url(sid, all_specs)
        if u:
            urls.append(u)
    return _dedupe(urls)[:max_refs]


def _refs_hero_only(
    target_spec: GoldenSpec, all_specs: list[GoldenSpec], max_refs: int = 4
) -> list[str]:
    if target_spec.image_url:
        return [target_spec.image_url]
    return []


def _refs_front_angles(
    target_spec: GoldenSpec, all_specs: list[GoldenSpec], max_refs: int = 4
) -> list[str]:
    urls: list[str] = []
    if target_spec.image_url:
        urls.append(target_spec.image_url)
    for sid in ["hero_front_left_45", "hero_front_right_45"]:
        if sid == target_spec.shot_id:
            continue
        u = _spec_url(sid, all_specs)
        if u:
            urls.append(u)
    return _dedupe(urls)[:max_refs]


REFERENCE_PACK_STRATEGIES: dict[str, Callable[..., list[str]]] = {
    "default": select_reference_urls,
    "label_heavy": _refs_label_heavy,
    "hero_only": _refs_hero_only,
    "front_angles": _refs_front_angles,
}

REFERENCE_PACK_STRATEGY_NAMES: frozenset[str] = frozenset(REFERENCE_PACK_STRATEGIES)


def select_references_with_strategy(
    target_spec: GoldenSpec,
    all_specs: list[GoldenSpec],
    max_refs: int = 4,
    strategy: str = "default",
) -> list[str]:
    fn = REFERENCE_PACK_STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown reference pack strategy {strategy!r}, "
            f"choose from {sorted(REFERENCE_PACK_STRATEGY_NAMES)}"
        )
    return fn(target_spec, all_specs, max_refs)
