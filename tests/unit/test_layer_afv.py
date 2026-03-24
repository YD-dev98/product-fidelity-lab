from __future__ import annotations

import json

from product_fidelity_lab.evaluation.layer_afv import _parse_verdicts
from product_fidelity_lab.models.golden_spec import AtomicFact


def _facts() -> list[AtomicFact]:
    return [
        AtomicFact(
            id="F1", category="GEOMETRY", fact="Bottle centered",
            critical=True, importance="high",
        ),
        AtomicFact(id="F2", category="COLOR", fact="Label is gold", importance="medium"),
        AtomicFact(id="F3", category="DETAIL", fact="Text is legible", importance="low"),
    ]


class TestParseVerdicts:
    def test_valid_json(self) -> None:
        response = json.dumps([
            {"id": "F1", "verdict": True, "confidence": 0.95, "reasoning": "Clearly centered"},
            {"id": "F2", "verdict": True, "confidence": 0.8, "reasoning": "Gold visible"},
            {"id": "F3", "verdict": False, "confidence": 0.6, "reasoning": "Blurry text"},
        ])
        verdicts = _parse_verdicts(response, _facts())
        assert len(verdicts) == 3
        assert verdicts[0].verdict is True
        assert verdicts[2].verdict is False

    def test_markdown_fenced_json(self) -> None:
        response = "```json\n" + json.dumps([
            {"id": "F1", "verdict": True, "confidence": 0.9, "reasoning": "ok"},
        ]) + "\n```"
        verdicts = _parse_verdicts(response, _facts())
        assert len(verdicts) == 1
        assert verdicts[0].verdict is True

    def test_malformed_json_raises(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="AFV parse failure"):
            _parse_verdicts("this is not json", _facts())

    def test_all_true(self) -> None:
        response = json.dumps([
            {"id": f"F{i+1}", "verdict": True, "confidence": 0.9, "reasoning": "ok"}
            for i in range(3)
        ])
        verdicts = _parse_verdicts(response, _facts())
        assert all(v.verdict is True for v in verdicts)

    def test_empty_response_raises(self) -> None:
        import pytest

        with pytest.raises(RuntimeError, match="AFV parse failure"):
            _parse_verdicts("", _facts())
