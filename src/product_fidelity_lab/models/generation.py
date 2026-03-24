"""Generation request and result models."""

from __future__ import annotations

from pydantic import BaseModel


class GenerationRequest(BaseModel):
    prompt: str
    reference_urls: list[str]
    image_size: str = "square_hd"
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    seed: int | None = None


class EditRequest(BaseModel):
    base_image_url: str
    reference_urls: list[str]
    prompt: str
    image_size: str = "square_hd"
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    seed: int | None = None


class GenerationResult(BaseModel):
    image_url: str
    seed: int
    model_id: str
    duration_ms: int
    cost_estimate: float
