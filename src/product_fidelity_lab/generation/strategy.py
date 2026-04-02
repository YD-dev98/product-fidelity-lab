"""Generation strategy routing and implementations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

import structlog

from product_fidelity_lab.generation.presets import compile_prompt
from product_fidelity_lab.models.generation import GenerationRequest, GenerationResult
from product_fidelity_lab.models.preset import Candidate, StudioPreset
from product_fidelity_lab.models.product import (
    ModelStatus,
    ProductModel,
    ProductProfile,
    ProviderMode,
)

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


FAL_LORA_INFERENCE = "fal-ai/flux-lora"
LORA_COST_PER_IMAGE = 0.06


class AdapterAssistedStrategy:
    """Generate using a trained LoRA adapter model."""

    def __init__(self, product_model: ProductModel) -> None:
        self._model = product_model

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
        prompt = compile_prompt(preset, profile, 0, style_prompt)
        effective_aspect = aspect_ratio_override or preset.aspect_ratio

        # Prepend trigger word to prompt
        if self._model.trigger_word:
            prompt = f"{self._model.trigger_word} {prompt}"

        candidates: list[Candidate] = []
        for i in range(num_candidates):
            seed = (base_seed + i) if base_seed is not None else None

            input_data: dict[str, object] = {
                "prompt": prompt,
                "lora_path": self._model.external_model_id,
                "lora_scale": self._model.strength,
                "image_size": _aspect_to_size(effective_aspect),
                "guidance_scale": preset.guidance_scale or 3.5,
                "num_inference_steps": preset.num_inference_steps or 28,
            }
            if seed is not None:
                input_data["seed"] = seed

            import time

            start = time.monotonic()
            result = await fal_client.subscribe(FAL_LORA_INFERENCE, input_data)
            duration_ms = int((time.monotonic() - start) * 1000)

            image_url = str(result["images"][0]["url"])  # type: ignore[index]
            actual_seed = int(result.get("seed", seed or 0))  # type: ignore[arg-type]

            candidates.append(
                Candidate(
                    candidate_id=uuid.uuid4().hex[:8],
                    image_url=image_url,
                    seed=actual_seed,
                    model_id=FAL_LORA_INFERENCE,
                    strategy_used="adapter_assisted",
                    generation_ms=duration_ms,
                    cost_estimate=LORA_COST_PER_IMAGE,
                )
            )
            logger.info(
                "strategy.adapter_candidate",
                candidate=i + 1,
                total=num_candidates,
                seed=actual_seed,
            )

        return candidates


def route_strategy(
    product: Product,
    preset: StudioPreset,
    product_model: ProductModel | None = None,
) -> GenerationStrategy:
    """Select the best generation strategy based on product model status."""
    if (
        product_model is not None
        and product_model.mode in (ProviderMode.ADAPTER, ProviderMode.FINETUNE)
        and product_model.status == ModelStatus.READY
        and product_model.external_model_id
    ):
        logger.info(
            "strategy.using_adapter",
            model_id=product_model.external_model_id[:60],
        )
        return AdapterAssistedStrategy(product_model)

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
