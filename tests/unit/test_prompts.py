from __future__ import annotations

import pytest

from product_fidelity_lab.generation.prompts import (
    PROMPT_STRATEGIES,
    PROMPT_STRATEGY_NAMES,
    REFERENCE_PACK_STRATEGIES,
    REFERENCE_PACK_STRATEGY_NAMES,
    build_generation_prompt,
    build_prompt_with_strategy,
    select_reference_urls,
    select_references_with_strategy,
)
from product_fidelity_lab.models.golden_spec import (
    ROI,
    AtomicFact,
    ExpectedText,
    GoldenSpec,
)

# -- Fixtures ----------------------------------------------------------------


def _make_spec(
    shot_id: str = "hero_front_straight",
    image_url: str | None = "https://example.com/hero.jpg",
) -> GoldenSpec:
    return GoldenSpec(
        shot_id=shot_id,
        image_path=f"training/{shot_id}.jpg",
        image_url=image_url,
        category="training",
        shot_type="hero",
        atomic_facts=[AtomicFact(id="f1", category="GEOMETRY", fact="Bottle is centered")],
        expected_texts=[ExpectedText(text="BRAND", critical=True)],
        brand_colors_hex=["#ffffff"],
        rois=[ROI(x=0.1, y=0.1, width=0.8, height=0.8, label="product")],
        description="front-facing hero shot",
    )


def _make_all_specs() -> list[GoldenSpec]:
    ids = [
        "hero_front_straight",
        "hero_front_left_45",
        "hero_front_right_45",
        "label_closeup",
        "cap_closeup",
        "glass_material_detail",
    ]
    return [_make_spec(sid, f"https://example.com/{sid}.jpg") for sid in ids]


# -- Prompt strategies -------------------------------------------------------


class TestPromptStrategies:
    def test_baseline_matches_original(self) -> None:
        spec = _make_spec()
        original = build_generation_prompt(spec, 4)
        via_strategy = build_prompt_with_strategy(spec, 4, strategy="baseline")
        assert original == via_strategy

    @pytest.mark.parametrize("strategy", sorted(PROMPT_STRATEGY_NAMES))
    def test_all_strategies_include_image_refs(self, strategy: str) -> None:
        spec = _make_spec()
        prompt = build_prompt_with_strategy(spec, 3, strategy=strategy)
        assert "@image1" in prompt
        assert "@image2" in prompt
        assert "@image3" in prompt

    @pytest.mark.parametrize("strategy", ["label_emphasis", "detailed_label"])
    def test_no_brand_or_alcohol_terms(self, strategy: str) -> None:
        spec = _make_spec()
        prompt = build_prompt_with_strategy(spec, 2, strategy=strategy).lower()
        for term in ["bacardi", "rum", "alcohol", "liquor", "spirit"]:
            assert term not in prompt, f"Strategy {strategy!r} contains forbidden term {term!r}"

    def test_label_emphasis_mentions_readable(self) -> None:
        spec = _make_spec()
        prompt = build_prompt_with_strategy(spec, 2, strategy="label_emphasis").lower()
        assert "readable" in prompt or "visible" in prompt

    def test_detailed_label_describes_layout(self) -> None:
        spec = _make_spec()
        prompt = build_prompt_with_strategy(spec, 2, strategy="detailed_label").lower()
        assert "label" in prompt
        assert "text" in prompt

    def test_reference_focused_is_shorter(self) -> None:
        spec = _make_spec()
        baseline = build_prompt_with_strategy(spec, 2, strategy="baseline")
        focused = build_prompt_with_strategy(spec, 2, strategy="reference_focused")
        assert len(focused) < len(baseline)

    def test_unknown_strategy_raises(self) -> None:
        spec = _make_spec()
        with pytest.raises(ValueError, match="Unknown prompt strategy"):
            build_prompt_with_strategy(spec, 1, strategy="nonexistent")

    def test_all_strategies_registered(self) -> None:
        assert len(PROMPT_STRATEGIES) == 4
        expected = {"baseline", "label_emphasis", "detailed_label", "reference_focused"}
        assert expected == PROMPT_STRATEGY_NAMES


# -- Reference pack strategies -----------------------------------------------


class TestReferencePackStrategies:
    def test_default_matches_original(self) -> None:
        spec = _make_spec()
        all_specs = _make_all_specs()
        original = select_reference_urls(spec, all_specs)
        via_strategy = select_references_with_strategy(spec, all_specs, strategy="default")
        assert original == via_strategy

    def test_label_heavy_includes_label_closeup(self) -> None:
        all_specs = _make_all_specs()
        spec = _make_spec()
        refs = select_references_with_strategy(spec, all_specs, strategy="label_heavy")
        label_url = "https://example.com/label_closeup.jpg"
        assert label_url in refs

    def test_hero_only_returns_single(self) -> None:
        spec = _make_spec()
        all_specs = _make_all_specs()
        refs = select_references_with_strategy(spec, all_specs, strategy="hero_only")
        assert len(refs) == 1
        assert refs[0] == spec.image_url

    def test_front_angles_includes_45_shots(self) -> None:
        spec = _make_spec()
        all_specs = _make_all_specs()
        refs = select_references_with_strategy(spec, all_specs, strategy="front_angles")
        assert "https://example.com/hero_front_left_45.jpg" in refs
        assert "https://example.com/hero_front_right_45.jpg" in refs

    def test_no_duplicates(self) -> None:
        spec = _make_spec()
        all_specs = _make_all_specs()
        for strategy in REFERENCE_PACK_STRATEGY_NAMES:
            refs = select_references_with_strategy(spec, all_specs, strategy=strategy)
            assert len(refs) == len(set(refs)), f"Duplicates in {strategy!r}: {refs}"

    def test_missing_image_url_handled(self) -> None:
        spec = _make_spec(image_url=None)
        all_specs = _make_all_specs()
        refs = select_references_with_strategy(spec, all_specs, strategy="hero_only")
        assert refs == []

    def test_unknown_strategy_raises(self) -> None:
        spec = _make_spec()
        with pytest.raises(ValueError, match="Unknown reference pack strategy"):
            select_references_with_strategy(spec, [], strategy="nonexistent")

    def test_all_strategies_registered(self) -> None:
        assert len(REFERENCE_PACK_STRATEGIES) == 4
        expected = {"default", "label_heavy", "hero_only", "front_angles"}
        assert expected == REFERENCE_PACK_STRATEGY_NAMES

    def test_respects_max_refs(self) -> None:
        spec = _make_spec()
        all_specs = _make_all_specs()
        for strategy in REFERENCE_PACK_STRATEGY_NAMES:
            refs = select_references_with_strategy(spec, all_specs, max_refs=2, strategy=strategy)
            assert len(refs) <= 2, (
                f"Strategy {strategy!r} returned {len(refs)} refs"
            )
