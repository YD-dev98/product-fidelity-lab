from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from product_fidelity_lab.benchmark_protocol import (
    BenchmarkProtocol,
    load_protocol,
)
from product_fidelity_lab.models.golden_spec import (
    ROI,
    AtomicFact,
    ExpectedText,
    GoldenSpec,
)


def _make_spec(
    shot_id: str, image_url: str = "https://example.com/img.jpg",
) -> GoldenSpec:
    return GoldenSpec(
        shot_id=shot_id,
        image_path=f"training/{shot_id}.jpg",
        image_url=image_url,
        category="training",
        shot_type="hero",
        atomic_facts=[
            AtomicFact(id="f1", category="GEOMETRY", fact="test"),
        ],
        expected_texts=[ExpectedText(text="BRAND", critical=True)],
        brand_colors_hex=["#ffffff"],
        rois=[ROI(x=0, y=0, width=1, height=1, label="product")],
        description="test",
    )


class TestBenchmarkProtocol:
    def test_validate_forbidden(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="hero_front_straight",
            forbidden_shot_ids=["hero_front_straight"],
        )
        with pytest.raises(ValueError, match="forbids"):
            proto.validate_references(["hero_front_straight"])

    def test_validate_not_in_allowed(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="hero_front_straight",
            allowed_reference_shot_ids=["label_closeup", "cap_closeup"],
        )
        with pytest.raises(ValueError, match="only allows"):
            proto.validate_references(["side_left"])

    def test_validate_allowed_passes(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="hero_front_straight",
            allowed_reference_shot_ids=["label_closeup", "cap_closeup"],
            forbidden_shot_ids=["hero_front_straight"],
        )
        # Should not raise
        proto.validate_references(["label_closeup", "cap_closeup"])

    def test_select_references(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="hero_front_straight",
            allowed_reference_shot_ids=[
                "label_closeup", "cap_closeup",
            ],
            forbidden_shot_ids=["hero_front_straight"],
        )
        specs = [
            _make_spec("label_closeup", "https://example.com/label.jpg"),
            _make_spec("cap_closeup", "https://example.com/cap.jpg"),
            _make_spec(
                "hero_front_straight", "https://example.com/hero.jpg",
            ),
        ]
        urls = proto.select_references(specs)
        assert "https://example.com/label.jpg" in urls
        assert "https://example.com/cap.jpg" in urls
        assert "https://example.com/hero.jpg" not in urls

    def test_select_references_rejects_forbidden(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="hero_front_straight",
            forbidden_shot_ids=["hero_front_straight"],
        )
        with pytest.raises(ValueError, match="forbids"):
            proto.select_references(
                [], shot_ids=["hero_front_straight"],
            )

    def test_respects_max_refs(self) -> None:
        proto = BenchmarkProtocol(
            benchmark_id="test",
            mode="held_out_synthesis",
            target_shot_id="x",
            allowed_reference_shot_ids=["a", "b", "c", "d"],
        )
        specs = [_make_spec(sid) for sid in ["a", "b", "c", "d"]]
        urls = proto.select_references(specs, max_refs=2)
        assert len(urls) == 2


class TestLoadProtocol:
    def test_round_trip(self) -> None:
        data = {
            "benchmark_id": "test",
            "mode": "held_out_synthesis",
            "target_shot_id": "hero_front_straight",
            "allowed_reference_shot_ids": ["label_closeup"],
            "forbidden_shot_ids": ["hero_front_straight"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(data, f)
            f.flush()
            proto = load_protocol(Path(f.name))
        assert proto.benchmark_id == "test"
        assert proto.forbidden_shot_ids == ["hero_front_straight"]

    def test_load_actual_benchmark(self) -> None:
        path = Path(
            "data/benchmarks/held_out_hero_front_straight.json",
        )
        if not path.exists():
            pytest.skip("benchmark file not present")
        proto = load_protocol(path)
        assert proto.target_shot_id == "hero_front_straight"
        assert "hero_front_straight" in proto.forbidden_shot_ids
        assert "hero_front_straight" not in proto.allowed_reference_shot_ids
