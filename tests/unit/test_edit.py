from __future__ import annotations

import pytest

from product_fidelity_lab.generation.edit import (
    build_edit_prompt,
    build_image_urls,
)
from product_fidelity_lab.models.generation import EditRequest

# -- EditRequest model -------------------------------------------------------


class TestEditRequest:
    def test_basic_construction(self) -> None:
        req = EditRequest(
            base_image_url="https://example.com/base.jpg",
            reference_urls=["https://example.com/label.jpg"],
            prompt="Fix the label.",
        )
        assert req.base_image_url == "https://example.com/base.jpg"
        assert req.guidance_scale == 3.5
        assert req.seed is None

    def test_defaults(self) -> None:
        req = EditRequest(
            base_image_url="https://example.com/base.jpg",
            reference_urls=[],
            prompt="test",
        )
        assert req.image_size == "square_hd"
        assert req.num_inference_steps == 28


# -- Image URL ordering (most likely costly bug) ----------------------------


class TestBuildImageUrls:
    def test_base_image_is_first(self) -> None:
        req = EditRequest(
            base_image_url="https://example.com/base.jpg",
            reference_urls=[
                "https://example.com/label.jpg",
                "https://example.com/hero.jpg",
            ],
            prompt="test",
        )
        urls = build_image_urls(req)
        assert urls[0] == "https://example.com/base.jpg"
        assert urls[1] == "https://example.com/label.jpg"
        assert urls[2] == "https://example.com/hero.jpg"
        assert len(urls) == 3

    def test_no_references(self) -> None:
        req = EditRequest(
            base_image_url="https://example.com/base.jpg",
            reference_urls=[],
            prompt="test",
        )
        urls = build_image_urls(req)
        assert urls == ["https://example.com/base.jpg"]


# -- Prompt role tags --------------------------------------------------------


class TestBuildEditPrompt:
    def test_adds_missing_tags(self) -> None:
        prompt = build_edit_prompt("Fix the label.", 3)
        assert "@image1" in prompt
        assert "@image2" in prompt
        assert "@image3" in prompt

    def test_preserves_existing_tags(self) -> None:
        prompt = build_edit_prompt(
            "@image1 is the base. Fix label from @image2.",
            3,
        )
        assert prompt.count("@image1") == 1
        assert prompt.count("@image2") == 1
        assert "@image3" in prompt

    def test_role_mapping(self) -> None:
        """@image1=base, @image2=label ref, @image3=hero ref."""
        prompt = (
            "@image1 is the base image. "
            "Match the label from @image2. "
            "Use @image3 as hero reference."
        )
        result = build_edit_prompt(prompt, 3)
        # Should not add duplicate tags
        assert result.count("@image1") == 1
        assert result.count("@image2") == 1
        assert result.count("@image3") == 1


# -- Prompt variant content --------------------------------------------------


class TestPromptVariants:
    """Test the actual prompt strings used in the experiment."""

    @pytest.fixture()
    def prompts(self) -> dict[str, str]:
        # Import inline to avoid coupling to script structure
        return {
            "preserve_and_fix": (
                "@image1 is the base image. Preserve the bottle shape, "
                "pose, cap, glass, lighting, and background exactly. "
                "Correct only the front label to match the label in "
                "@image2 and @image3. The label text must be sharp and "
                "readable. Do not change bottle geometry, cap, glass "
                "color, liquid, camera angle, background, lighting, or "
                "shadow. Only replace the front label artwork and text."
            ),
            "label_transfer": (
                "Edit the front label on the product in @image1 to "
                "exactly match the label from @image2. Use @image3 as "
                "overall product reference. Keep everything else — "
                "bottle, cap, glass, liquid, lighting, background, "
                "shadow — completely unchanged. Only replace the front "
                "label artwork and text."
            ),
            "branded_minimal": (
                "@image1 is the base image. The front label should read "
                "BACARDI SUPERIOR with a bat logo. Match the label layout "
                "from @image2. Do not change bottle geometry, cap, glass "
                "color, liquid, camera angle, background, lighting, or "
                "shadow."
            ),
            "branded_reference": (
                "@image1 is the base image. The front label shows the "
                "brand name and product line exactly as printed on the "
                "label in @image2. Copy all text, logo, and layout from "
                "@image2 onto the label area of @image1. Do not change "
                "bottle geometry, cap, glass color, liquid, camera angle, "
                "background, lighting, or shadow."
            ),
        }

    @pytest.mark.parametrize("name", ["preserve_and_fix", "label_transfer"])
    def test_non_branded_no_alcohol_terms(
        self, name: str, prompts: dict[str, str],
    ) -> None:
        text = prompts[name].lower()
        for term in ["bacardi", "rum", "alcohol", "liquor", "spirit"]:
            assert term not in text, (
                f"Non-branded variant {name!r} contains {term!r}"
            )

    @pytest.mark.parametrize("name", ["branded_minimal", "branded_reference"])
    def test_branded_contains_brand_signal(
        self, name: str, prompts: dict[str, str],
    ) -> None:
        text = prompts[name].lower()
        assert "bacardi" in text or "brand" in text

    @pytest.mark.parametrize("name", [
        "preserve_and_fix", "label_transfer",
        "branded_minimal", "branded_reference",
    ])
    def test_all_variants_reference_image1(
        self, name: str, prompts: dict[str, str],
    ) -> None:
        assert "@image1" in prompts[name]

    @pytest.mark.parametrize("name", [
        "preserve_and_fix", "label_transfer",
        "branded_minimal", "branded_reference",
    ])
    def test_all_variants_reference_image2(
        self, name: str, prompts: dict[str, str],
    ) -> None:
        assert "@image2" in prompts[name]
