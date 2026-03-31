from __future__ import annotations

import pytest
from pydantic import ValidationError

from product_fidelity_lab.models.preset import RenderRequest


class TestRenderRequest:
    def test_valid_request(self) -> None:
        req = RenderRequest(
            product_id="p1",
            preset_id="clean_packshot",
        )
        assert req.num_candidates == 4
        assert req.skip_repair is False
        assert req.skip_ranking is False

    def test_custom_candidates(self) -> None:
        req = RenderRequest(
            product_id="p1",
            preset_id="clean_packshot",
            num_candidates=8,
            seed=42,
            style_prompt="dramatic lighting",
        )
        assert req.num_candidates == 8
        assert req.seed == 42
        assert req.style_prompt == "dramatic lighting"

    def test_candidates_min_bound(self) -> None:
        with pytest.raises(ValidationError):
            RenderRequest(
                product_id="p1",
                preset_id="clean_packshot",
                num_candidates=0,
            )

    def test_candidates_max_bound(self) -> None:
        with pytest.raises(ValidationError):
            RenderRequest(
                product_id="p1",
                preset_id="clean_packshot",
                num_candidates=13,
            )

    def test_aspect_ratio_override(self) -> None:
        req = RenderRequest(
            product_id="p1",
            preset_id="clean_packshot",
            aspect_ratio="16:9",
        )
        assert req.aspect_ratio == "16:9"
