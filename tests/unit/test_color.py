from __future__ import annotations

import numpy as np
from PIL import Image

from product_fidelity_lab.evaluation.color import (
    compare_to_brand_colors,
    extract_dominant_colors,
    hex_to_rgb,
    rgb_to_hex,
)
from product_fidelity_lab.models.golden_spec import ROI


class TestColorConversion:
    def test_hex_to_rgb(self) -> None:
        rgb = hex_to_rgb("#ff8000")
        assert rgb[0] == 255
        assert rgb[1] == 128
        assert rgb[2] == 0

    def test_rgb_to_hex(self) -> None:
        h = rgb_to_hex(np.array([255.0, 128.0, 0.0]))
        assert h == "#ff8000"


class TestExtractDominantColors:
    def test_solid_color(self) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        colors = extract_dominant_colors(img, n_colors=1)
        assert len(colors) == 1
        # LAB: red should have high a* value
        assert colors[0][1] > 50  # a* channel

    def test_with_roi(self) -> None:
        # Left half red, right half blue
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        for x in range(50, 100):
            for y in range(100):
                img.putpixel((x, y), (0, 0, 255))

        roi = ROI(x=0.0, y=0.0, width=0.5, height=1.0, label="product")
        colors = extract_dominant_colors(img, roi=roi, n_colors=1)
        # Should extract red from the ROI (left half)
        assert colors[0][1] > 50  # a* positive = red


class TestCompareColors:
    def test_identical_colors(self) -> None:
        img = Image.new("RGB", (100, 100), color=(200, 160, 80))
        extracted = extract_dominant_colors(img, n_colors=1)
        score, pairs = compare_to_brand_colors(extracted, ["#c8a050"])
        assert score > 0.8
        assert len(pairs) == 1

    def test_very_different_colors(self) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        extracted = extract_dominant_colors(img, n_colors=1)
        score, pairs = compare_to_brand_colors(extracted, ["#0000ff"])
        assert score < 0.5

    def test_empty_brand_colors(self) -> None:
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        extracted = extract_dominant_colors(img, n_colors=1)
        score, pairs = compare_to_brand_colors(extracted, [])
        assert score == 1.0
        assert pairs == []
