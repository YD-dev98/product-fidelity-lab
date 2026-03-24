from __future__ import annotations

import numpy as np

from product_fidelity_lab.evaluation.layer_depth import compare_depth
from product_fidelity_lab.models.golden_spec import ROI


class TestCompareDepth:
    def test_identical_maps(self) -> None:
        depth = np.random.default_rng(42).random((64, 64))
        score = compare_depth(depth, depth.copy())
        assert score.ssim > 0.99
        assert score.combined > 0.95

    def test_inverted_maps(self) -> None:
        depth = np.random.default_rng(42).random((64, 64))
        inverted = 1.0 - depth
        score = compare_depth(depth, inverted)
        assert score.ssim < 0.5
        assert score.correlation < 0.0

    def test_with_roi(self) -> None:
        depth = np.random.default_rng(42).random((100, 100))
        roi = ROI(x=0.2, y=0.2, width=0.5, height=0.5, label="product")
        score = compare_depth(depth, depth.copy(), roi=roi)
        assert score.roi_used is True
        assert score.ssim > 0.99

    def test_different_sizes(self) -> None:
        golden = np.random.default_rng(42).random((64, 64))
        generated = np.random.default_rng(42).random((128, 128))
        score = compare_depth(golden, generated)
        # Should resize and still compute
        assert 0.0 <= score.combined <= 1.0

    def test_flat_map(self) -> None:
        flat = np.ones((64, 64)) * 0.5
        score = compare_depth(flat, flat.copy())
        assert score.mse == 0.0
