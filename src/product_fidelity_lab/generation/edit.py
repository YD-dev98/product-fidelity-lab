"""Image editing via fal.ai FLUX.2 [flex] edit endpoint."""

from __future__ import annotations

import time
from typing import Any

import structlog

from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.generation import EditRequest, GenerationResult

logger = structlog.get_logger()

FLUX_EDIT_MODEL = "fal-ai/flux-2-flex/edit"
COST_PER_IMAGE_ESTIMATED = 0.06  # estimated, not confirmed


def build_edit_prompt(base_prompt: str, n_images: int) -> str:
    """Ensure @imageN tags are present for all images in the edit call."""
    parts = [base_prompt]
    for i in range(1, n_images + 1):
        tag = f"@image{i}"
        if tag not in base_prompt:
            parts.append(tag)
    return " ".join(parts)


def build_image_urls(request: EditRequest) -> list[str]:
    """Build image_urls list: base image first, then references."""
    return [request.base_image_url, *request.reference_urls]


async def edit_image(
    request: EditRequest,
    fal_client: FalClient,
) -> GenerationResult:
    """Edit an image using FLUX.2 flex edit endpoint.

    The base image is @image1, references are @image2, @image3, etc.
    """
    image_urls = build_image_urls(request)
    prompt = build_edit_prompt(request.prompt, len(image_urls))

    input_data: dict[str, object] = {
        "prompt": prompt,
        "image_urls": image_urls,
        "image_size": request.image_size,
        "guidance_scale": request.guidance_scale,
        "num_inference_steps": request.num_inference_steps,
    }
    if request.seed is not None:
        input_data["seed"] = request.seed

    start = time.monotonic()
    result: dict[str, Any] = await fal_client.subscribe(
        FLUX_EDIT_MODEL, input_data,
    )
    duration_ms = int((time.monotonic() - start) * 1000)

    image_url = str(result["images"][0]["url"])
    seed = int(result.get("seed", 0))

    logger.info(
        "edit.complete",
        model=FLUX_EDIT_MODEL,
        duration_ms=duration_ms,
        seed=seed,
    )

    return GenerationResult(
        image_url=image_url,
        seed=seed,
        model_id=FLUX_EDIT_MODEL,
        duration_ms=duration_ms,
        cost_estimate=COST_PER_IMAGE_ESTIMATED,
    )
