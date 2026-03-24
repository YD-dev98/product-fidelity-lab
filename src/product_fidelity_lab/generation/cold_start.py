"""Cold start generation via fal.ai FLUX.2 [flex]."""

from __future__ import annotations

import time

import structlog

from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.generation import GenerationRequest, GenerationResult

logger = structlog.get_logger()

FLUX_MODEL = "fal-ai/flux-2-flex"
COST_PER_IMAGE = 0.05


def build_prompt(base_prompt: str, reference_urls: list[str]) -> str:
    """Rewrite prompt to include @imageN references for multi-reference."""
    parts = [base_prompt]
    for i, _url in enumerate(reference_urls, 1):
        tag = f"@image{i}"
        if tag not in base_prompt:
            parts.append(tag)
    return " ".join(parts)


async def cold_generate(
    request: GenerationRequest,
    fal_client: FalClient,
    cache: object | None = None,
) -> GenerationResult:
    """Generate a single image using FLUX.2 flex with multi-reference.

    Uses ResultCache if provided for deterministic replay of seeded generations.
    """
    from typing import Any

    prompt = build_prompt(request.prompt, request.reference_urls)

    input_data: dict[str, object] = {
        "prompt": prompt,
        "image_urls": request.reference_urls,
        "image_size": request.image_size,
        "guidance_scale": request.guidance_scale,
        "num_inference_steps": request.num_inference_steps,
    }
    if request.seed is not None:
        input_data["seed"] = request.seed

    async def _call_flux() -> dict[str, Any]:
        return await fal_client.subscribe(FLUX_MODEL, input_data)

    cache_inputs: dict[str, Any] = dict(input_data)  # type: ignore[arg-type]
    start = time.monotonic()
    if cache is not None:
        entry = await cache.get_or_compute(cache_inputs, _call_flux)  # type: ignore[union-attr]
        result: dict[str, Any] = entry["result"]  # type: ignore[reportUnknownVariableType]
    else:
        result = await _call_flux()
    duration_ms = int((time.monotonic() - start) * 1000)

    image_url = str(result["images"][0]["url"])  # type: ignore[reportUnknownArgumentType]
    seed = int(result.get("seed", 0))  # type: ignore[reportUnknownArgumentType]

    logger.info(
        "generation.complete",
        model=FLUX_MODEL,
        duration_ms=duration_ms,
        seed=seed,
    )

    return GenerationResult(
        image_url=image_url,
        seed=seed,
        model_id=FLUX_MODEL,
        duration_ms=duration_ms,
        cost_estimate=COST_PER_IMAGE,
    )
