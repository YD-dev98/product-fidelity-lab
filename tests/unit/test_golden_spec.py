from __future__ import annotations

import json
from pathlib import Path

import pytest

from product_fidelity_lab.evaluation.spec_loader import load_all_specs, load_spec, validate_specs
from product_fidelity_lab.models.golden_spec import ROI, ExpectedText, GoldenSpec


def _make_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "shot_id": "hero_front",
        "image_path": "training/hero.jpg",
        "category": "training",
        "shot_type": "hero",
        "atomic_facts": [
            {
                "id": "F1", "category": "GEOMETRY",
                "fact": "Bottle is centered", "critical": True,
                "importance": "high",
            },
            {"id": "F2", "category": "COLOR", "fact": "Label is gold", "importance": "medium"},
        ],
        "expected_texts": [
            {"text": "Macallan", "critical": True, "match_mode": "exact_token"},
        ],
        "brand_colors_hex": ["#c8a951"],
        "rois": [
            {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8, "label": "product"},
        ],
        "description": "Hero front-left shot of bottle",
    }
    base.update(overrides)
    return base


class TestGoldenSpec:
    def test_valid_spec(self) -> None:
        spec = GoldenSpec.model_validate(_make_spec())
        assert spec.shot_id == "hero_front"
        assert len(spec.atomic_facts) == 2
        assert spec.atomic_facts[0].critical is True

    def test_critical_text_requires_exact(self) -> None:
        with pytest.raises(ValueError, match="critical=True requires"):
            ExpectedText(text="Macallan", critical=True, match_mode="fuzzy")

    def test_non_critical_fuzzy_ok(self) -> None:
        et = ExpectedText(text="tagline", critical=False, match_mode="fuzzy")
        assert et.match_mode == "fuzzy"

    def test_roi_normalization(self) -> None:
        roi = ROI(x=0.1, y=0.2, width=0.5, height=0.6, label="product")
        assert 0 <= roi.x <= 1
        assert 0 <= roi.x + roi.width <= 1


class TestSpecLoader:
    def test_load_spec(self, tmp_path: Path) -> None:
        spec_data = _make_spec()
        spec_path = tmp_path / "hero_front.json"
        spec_path.write_text(json.dumps(spec_data))

        spec = load_spec(spec_path)
        assert spec.shot_id == "hero_front"

    def test_load_all_specs(self, tmp_path: Path) -> None:
        for i in range(3):
            data = _make_spec(shot_id=f"shot_{i}")
            (tmp_path / f"shot_{i}.json").write_text(json.dumps(data))

        specs = load_all_specs(tmp_path)
        assert len(specs) == 3

    def test_validate_missing_image(self, tmp_path: Path) -> None:
        spec = GoldenSpec.model_validate(_make_spec())
        errors = validate_specs([spec], tmp_path)
        assert len(errors) == 1
        assert "hero.jpg" in errors[0]

    def test_validate_existing_image(self, tmp_path: Path) -> None:
        (tmp_path / "training").mkdir()
        (tmp_path / "training" / "hero.jpg").write_bytes(b"fake")
        spec = GoldenSpec.model_validate(_make_spec())
        errors = validate_specs([spec], tmp_path)
        assert len(errors) == 0
