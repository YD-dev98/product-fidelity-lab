"""Tests for product evaluation: spec synthesis and evaluation wrapper."""

from __future__ import annotations

from product_fidelity_lab.evaluation.product_eval import synthesize_spec
from product_fidelity_lab.models.product import PackagingType, ProductProfile


def _make_profile(**kwargs: object) -> ProductProfile:
    defaults = {"product_id": "p1"}
    defaults.update(kwargs)
    return ProductProfile(**defaults)  # type: ignore[arg-type]


class TestSynthesizeSpec:
    def test_basic_spec_from_profile(self) -> None:
        profile = _make_profile(
            packaging_type=PackagingType.BOTTLE,
            critical_texts=["BACARDI"],
            brand_colors_hex=["#cc1e1e"],
        )
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        assert spec.shot_id == "product_p1"
        assert spec.category == "validation"
        assert len(spec.atomic_facts) >= 3  # packaging + text + color
        assert len(spec.expected_texts) == 1
        assert spec.expected_texts[0].text == "BACARDI"
        assert spec.expected_texts[0].critical is True
        assert spec.brand_colors_hex == ["#cc1e1e"]
        assert spec.rois == []

    def test_critical_texts_become_critical_facts(self) -> None:
        profile = _make_profile(
            critical_texts=["BACARDI", "SUPERIOR"],
        )
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        text_facts = [f for f in spec.atomic_facts if f.category == "TEXT"]
        critical_facts = [f for f in text_facts if f.critical]
        assert len(critical_facts) == 2

    def test_brand_texts_non_critical(self) -> None:
        profile = _make_profile(
            brand_texts=["RUM", "750ml"],
            critical_texts=["RUM"],
        )
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        text_facts = [f for f in spec.atomic_facts if f.category == "TEXT"]
        non_critical = [f for f in text_facts if not f.critical]
        # "750ml" should be non-critical, "RUM" should be critical
        assert any("750ml" in f.fact for f in non_critical)

    def test_empty_profile_gets_minimal_spec(self) -> None:
        profile = _make_profile()
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        assert len(spec.atomic_facts) >= 1
        assert spec.expected_texts == []

    def test_material_summary_becomes_fact(self) -> None:
        profile = _make_profile(material_summary="glass")
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        material_facts = [
            f for f in spec.atomic_facts if f.category == "MATERIAL"
        ]
        assert len(material_facts) == 1
        assert "glass" in material_facts[0].fact

    def test_color_facts_limited_to_three(self) -> None:
        profile = _make_profile(
            brand_colors_hex=["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff"],
        )
        spec = synthesize_spec(profile, "https://example.com/img.jpg")

        color_facts = [f for f in spec.atomic_facts if f.category == "COLOR"]
        assert len(color_facts) == 3  # capped at 3

    def test_spec_image_url_set(self) -> None:
        profile = _make_profile()
        url = "https://example.com/rendered.jpg"
        spec = synthesize_spec(profile, url)
        assert spec.image_url == url
