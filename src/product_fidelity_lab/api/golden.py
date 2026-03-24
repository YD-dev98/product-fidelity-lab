"""Golden spec API endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.evaluation.spec_loader import load_all_specs

logger = structlog.get_logger()
router = APIRouter(prefix="/api", tags=["golden"])


@router.get("/golden/specs")
async def list_specs() -> list[dict[str, Any]]:
    """List all available golden specs."""
    settings = get_settings()
    specs_dir = settings.data_dir / "golden" / "specs"
    if not specs_dir.exists():
        return []
    specs = load_all_specs(specs_dir)
    return [s.model_dump() for s in specs]


class BootstrapRequest(BaseModel):
    image_url: str


@router.post("/golden/bootstrap")
async def bootstrap_spec(req: BootstrapRequest) -> dict[str, Any]:
    """Generate a draft golden spec from an image. Dev tool."""
    settings = get_settings()
    from product_fidelity_lab.evaluation.layer_afv import bootstrap_facts

    result = await bootstrap_facts(
        req.image_url,
        api_key=settings.gemini_api_key,
    )
    return result
