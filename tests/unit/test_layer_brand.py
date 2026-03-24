from __future__ import annotations

from product_fidelity_lab.evaluation.layer_brand import _is_contiguous_subsequence, _tokenize, compare_text
from product_fidelity_lab.models.golden_spec import ExpectedText


class TestTokenize:
    def test_basic(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self) -> None:
        assert _tokenize("Single Malt Scotch Whisky.") == ["single", "malt", "scotch", "whisky"]


class TestContiguousSubsequence:
    def test_present(self) -> None:
        haystack = ["single", "malt", "scotch", "whisky"]
        assert _is_contiguous_subsequence(["malt", "scotch"], haystack)

    def test_absent(self) -> None:
        haystack = ["single", "malt", "scotch", "whisky"]
        assert not _is_contiguous_subsequence(["malt", "whisky"], haystack)

    def test_empty_needle(self) -> None:
        assert _is_contiguous_subsequence([], ["a", "b"])

    def test_exact_match(self) -> None:
        assert _is_contiguous_subsequence(["a"], ["a"])


class TestCompareText:
    def test_exact_match(self) -> None:
        expected = [ExpectedText(text="Macallan", critical=True)]
        result = compare_text(expected, ["The Macallan Single Malt"])
        assert result.score == 1.0
        assert result.matches[0].matched is True
        assert len(result.critical_failures) == 0

    def test_exact_missing_triggers_critical(self) -> None:
        expected = [ExpectedText(text="Macallan", critical=True)]
        result = compare_text(expected, ["Some other brand"])
        assert result.matches[0].matched is False
        assert len(result.critical_failures) == 1
        assert "Macallan" in result.critical_failures[0]

    def test_fuzzy_match(self) -> None:
        expected = [ExpectedText(
            text="Premium Quality Spirits", critical=False, match_mode="fuzzy",
        )]
        result = compare_text(expected, ["Premium Quality Spirit"])
        assert result.matches[0].score > 0.8

    def test_empty_expected(self) -> None:
        result = compare_text([], ["random text"])
        assert result.score == 1.0

    def test_multi_line_ocr(self) -> None:
        """OCR text split across lines should still match phrases."""
        expected = [ExpectedText(text="Single Malt Scotch Whisky", critical=True)]
        extracted = ["The Macallan", "Single Malt", "Scotch Whisky"]
        result = compare_text(expected, extracted)
        assert result.matches[0].matched is True

    def test_non_critical_miss_no_gate(self) -> None:
        expected = [ExpectedText(text="tagline", critical=False)]
        result = compare_text(expected, ["no match here"])
        assert len(result.critical_failures) == 0
