from __future__ import annotations

from product_fidelity_lab.generation.cold_start import build_prompt


class TestBuildPrompt:
    def test_adds_image_refs(self) -> None:
        prompt = build_prompt(
            "Product on white sweep",
            ["http://example.com/a.jpg", "http://example.com/b.jpg"],
        )
        assert "@image1" in prompt
        assert "@image2" in prompt

    def test_no_duplicate_refs(self) -> None:
        prompt = build_prompt(
            "Product @image1 on white sweep",
            ["http://example.com/a.jpg"],
        )
        assert prompt.count("@image1") == 1

    def test_no_refs(self) -> None:
        prompt = build_prompt("Simple prompt", [])
        assert prompt == "Simple prompt"
