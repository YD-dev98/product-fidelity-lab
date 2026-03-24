from __future__ import annotations

from pathlib import Path

from product_fidelity_lab.evaluation.calibration import (
    compute_distribution,
    compute_human_correlation,
    compute_thresholds,
    load_thresholds,
    save_thresholds,
)


class TestDistribution:
    def test_basic_stats(self) -> None:
        dist = compute_distribution([0.8, 0.85, 0.9, 0.95, 1.0])
        assert abs(dist.mean - 0.9) < 0.01
        assert dist.min == 0.8
        assert dist.max == 1.0
        assert dist.n == 5

    def test_single_value(self) -> None:
        dist = compute_distribution([0.5])
        assert dist.mean == 0.5
        assert dist.std == 0.0


class TestThresholds:
    def test_compute_from_distribution(self) -> None:
        from product_fidelity_lab.evaluation.calibration import LayerDistribution

        dists = {
            "overall": LayerDistribution(mean=0.9, std=0.05, min=0.8, max=1.0, n=15)
        }
        thresholds = compute_thresholds(dists)
        assert thresholds["A"] > thresholds["B"] > thresholds["C"] > thresholds["D"]

    def test_no_overall_uses_defaults(self) -> None:
        thresholds = compute_thresholds({})
        assert thresholds == {"A": 0.85, "B": 0.70, "C": 0.55, "D": 0.40}

    def test_save_and_load(self, tmp_path: Path) -> None:
        thresholds = {"A": 0.87, "B": 0.72, "C": 0.57, "D": 0.42}
        path = tmp_path / "calibration" / "thresholds.json"
        save_thresholds(thresholds, path)
        loaded = load_thresholds(path)
        assert loaded == thresholds


class TestHumanCorrelation:
    def test_perfect_correlation(self) -> None:
        scores = [0.3, 0.5, 0.9]
        labels = ["fail", "borderline", "pass"]
        result = compute_human_correlation(scores, labels)
        assert result["spearman_rho"] > 0.9

    def test_confusion_matrix(self) -> None:
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = ["pass", "pass", "fail", "fail"]
        result = compute_human_correlation(scores, labels)
        cm = result["confusion_matrix"]
        assert cm["tp"] == 2
        assert cm["tn"] == 2
        assert cm["fp"] == 0
        assert cm["fn"] == 0

    def test_borderline_excluded_from_confusion(self) -> None:
        scores = [0.9, 0.68, 0.3]
        labels = ["pass", "borderline", "fail"]
        result = compute_human_correlation(scores, labels)
        cm = result["confusion_matrix"]
        assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == 2  # borderline excluded

    def test_borderline_near_threshold(self) -> None:
        scores = [0.69, 0.72]
        labels = ["borderline", "borderline"]
        result = compute_human_correlation(scores, labels)
        assert result["borderline_agreement_rate"] == 1.0
