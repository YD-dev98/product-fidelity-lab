from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from product_fidelity_lab.evaluation.perturbation import PerturbationType, apply_perturbation
from product_fidelity_lab.models.golden_spec import ROI


def _create_test_image(path: Path) -> Path:
    img = Image.new("RGB", (200, 200), color=(128, 64, 32))
    # Add some text-like contrast in center
    for x in range(80, 120):
        for y in range(80, 120):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path)
    return path


class TestPerturbations:
    def test_blur_changes_image(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        original = np.array(Image.open(src))
        result = apply_perturbation(src, PerturbationType.BLUR, {"sigma": 3})
        result_arr = np.array(result)
        assert not np.array_equal(original, result_arr)

    def test_blur_reduces_sharpness(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        original = np.array(Image.open(src), dtype=np.float64)
        result = np.array(
            apply_perturbation(src, PerturbationType.BLUR, {"sigma": 5}),
            dtype=np.float64,
        )
        # Laplacian variance as sharpness proxy
        from scipy.ndimage import laplace
        orig_sharp = np.var(laplace(original.mean(axis=2)))
        blur_sharp = np.var(laplace(result.mean(axis=2)))
        assert blur_sharp < orig_sharp

    def test_crop_shift(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        result = apply_perturbation(src, PerturbationType.CROP_SHIFT, {"offset_pct": 0.1})
        assert result.size == Image.open(src).size

    def test_hue_rotation(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        original = np.array(Image.open(src))
        result = np.array(apply_perturbation(src, PerturbationType.HUE_ROTATION, {"degrees": 30}))
        assert not np.array_equal(original, result)

    def test_text_removal(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        rois = [ROI(x=0.3, y=0.3, width=0.4, height=0.4, label="label")]
        result = np.array(apply_perturbation(src, PerturbationType.TEXT_REMOVAL, rois=rois))
        original = np.array(Image.open(src))
        # The center white area should be filled
        center_original = original[80:120, 80:120]
        center_result = result[80:120, 80:120]
        assert not np.array_equal(center_original, center_result)

    def test_brightness_reduction(self, tmp_path: Path) -> None:
        src = _create_test_image(tmp_path / "src.jpg")
        original = np.array(Image.open(src), dtype=np.float64)
        result = np.array(
            apply_perturbation(src, PerturbationType.BRIGHTNESS_REDUCTION, {"factor": 0.4}),
            dtype=np.float64,
        )
        assert result.mean() < original.mean()
