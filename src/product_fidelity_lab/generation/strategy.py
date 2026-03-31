"""Generation strategy routing and implementations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

import structlog

from product_fidelity_lab.generation.presets import compile_prompt
from product_fidelity_lab.models.generation import GenerationRequest, GenerationResult
from product_fidelity_lab.models.preset import Candidate, StudioPreset
from product_fidelity_lab.models.product import ProductProfile

if TYPE_CHECKING:
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.models.product import Product

logger = structlog.get_logger()


class GenerationStrategy(Protocol):
    async def generate_candidates(
        self,
        profile: ProductProfile,
        preset: StudioPreset,
        reference_urls: list[str],
        num_candidates: int,
        fal_client: FalClient,
        style_prompt: str = "",
        base_seed: int | None = None,
        aspect_ratio_override: str | None = None,
    ) -> list[Candidate]: ...


class ReferenceOnlyStrategy:
    """Generate using product reference images only via cold_generate."""

    async def generate_candidates(
        self,
        profile: ProductProfile,
        preset: StudioPreset,
        reference_urls: list[str],
        num_candidates: int,
        fal_client: FalClient,
        style_prompt: str = "",
        base_seed: int | None = None,
        aspect_ratio_override: str | None = None,
    ) -> list[Candidate]:
        from product_fidelity_lab.generation.cold_start import cold_generate

        prompt = compile_prompt(preset, profile, len(reference_urls), style_prompt)
        effective_aspect = aspect_ratio_override or preset.aspect_ratio

        candidates: list[Candidate] = []
        for i in range(num_candidates):
            seed = (base_seed + i) if base_seed is not None else None

            request = GenerationRequest(
                prompt=prompt,
                reference_urls=reference_urls,
                image_size=_aspect_to_size(effective_aspect),
                guidance_scale=preset.guidance_scale or 3.5,
                num_inference_steps=preset.num_inference_steps or 28,
                seed=seed,
            )

            result: GenerationResult = await cold_generate(request, fal_client)

            candidates.append(
                Candidate(
                    candidate_id=uuid.uuid4().hex[:8],
                    image_url=result.image_url,
                    seed=result.seed,
                    model_id=result.model_id,
                    strategy_used="reference_only",
                    generation_ms=result.duration_ms,
                    cost_estimate=result.cost_estimate,
                )
            )
            logger.info(
                "strategy.candidate_generated",
                candidate=i + 1,
                total=num_candidates,
                seed=result.seed,
            )

        return candidates


def route_strategy(
    product: Product,
    preset: StudioPreset,
) -> GenerationStrategy:
    """Select the best generation strategy based on product and preset."""
    # Phase 1: always reference-only
    return ReferenceOnlyStrategy()


def _aspect_to_size(aspect_ratio: str) -> str:
    """Map aspect ratio string to fal.ai image_size value."""
    mapping = {
        "1:1": "square_hd",
        "4:5": "portrait_4_3",
        "16:9": "landscape_16_9",
        "9:16": "portrait_16_9",
        "4:3": "landscape_4_3",
        "3:4": "portrait_4_3",
    }
    return mapping.get(aspect_ratio, "square_hd")
