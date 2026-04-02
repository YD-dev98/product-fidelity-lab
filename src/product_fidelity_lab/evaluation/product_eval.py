"""Product-oriented evaluation wrapper.

Builds a synthetic evaluation spec from a ProductProfile and runs
brand + AFV checks (no depth). This is the opt-in deep QA path,
separate from the lightweight _run_qa in the render pipeline.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from product_fidelity_lab.evaluation.aggregator import compute_grade
from product_fidelity_lab.evaluation.layer_afv import verify_facts
from product_fidelity_lab.evaluation.layer_brand import evaluate_brand
from product_fidelity_lab.models.evaluation import (
    AFVReport,
    BrandReport,
)
from product_fidelity_lab.models.golden_spec import (
    AtomicFact,
    ExpectedText,
    GoldenSpec,
)
from product_fidelity_lab.models.product import ProductProfile

logger = structlog.get_logger()


def synthesize_spec(
    profile: ProductProfile,
    image_url: str,
) -> GoldenSpec:
    """Build a synthetic GoldenSpec from a ProductProfile.

    Generates atomic facts from profile fields (packaging type, material,
    brand texts) and converts critical_texts to ExpectedText entries.
    No depth map or ROIs — those require golden reference data.
    """
    facts: list[AtomicFact] = []
    fact_id = 1

    # Packaging type fact
    if profile.packaging_type.value != "unknown":
        facts.append(AtomicFact(
            id=f"F{fact_id}",
            category="GEOMETRY",
            fact=f"The product is a {profile.packaging_type.value}",
            critical=False,
            importance="high",
        ))
        fact_id += 1

    # Material summary fact
    if profile.material_summary:
        facts.append(AtomicFact(
            id=f"F{fact_id}",
            category="MATERIAL",
            fact=f"The product material is {profile.material_summary}",
            critical=False,
            importance="medium",
        ))
        fact_id += 1

    # Brand text facts (critical)
    for text in profile.critical_texts:
        facts.append(AtomicFact(
            id=f"F{fact_id}",
            category="TEXT",
            fact=f'The text "{text}" is visible on the product',
            critical=True,
            importance="high",
        ))
        fact_id += 1

    # Brand text facts (non-critical)
    for text in profile.brand_texts:
        if text not in profile.critical_texts:
            facts.append(AtomicFact(
                id=f"F{fact_id}",
                category="TEXT",
                fact=f'The text "{text}" appears on the product',
                critical=False,
                importance="medium",
            ))
            fact_id += 1

    # Color facts
    for hex_color in profile.brand_colors_hex[:3]:
        facts.append(AtomicFact(
            id=f"F{fact_id}",
            category="COLOR",
            fact=f"The brand color {hex_color} is present in the image",
            critical=False,
            importance="medium",
        ))
        fact_id += 1

    # If no facts at all, add a minimal one
    if not facts:
        facts.append(AtomicFact(
            id="F1",
            category="GEOMETRY",
            fact="A product is clearly visible in the image",
            critical=False,
            importance="high",
        ))

    # Build expected texts from critical_texts
    expected_texts = [
        ExpectedText(text=t, critical=True, match_mode="exact_token")
        for t in profile.critical_texts
    ]

    return GoldenSpec(
        shot_id=f"product_{profile.product_id}",
        image_path="",
        image_url=image_url,
        category="validation",
        shot_type="product_render",
        atomic_facts=facts,
        expected_texts=expected_texts,
        brand_colors_hex=profile.brand_colors_hex,
        rois=[],
        description=f"Synthetic spec for product {profile.product_id}",
    )


async def run_product_evaluation(
    image_url: str,
    profile: ProductProfile,
    *,
    fal_client: Any,
    gemini_api_key: str,
    gemini_model: str,
) -> dict[str, Any]:
    """Run brand + AFV evaluation against a product profile.

    Skips depth (no golden depth map available). Returns a result dict
    compatible with RunStore.update_result().
    """
    start = time.monotonic()

    spec = synthesize_spec(profile, image_url)

    # Run AFV + brand concurrently (no depth)
    afv_task = asyncio.create_task(_run_afv(
        image_url, spec, gemini_api_key, gemini_model,
    ))
    brand_task = asyncio.create_task(_run_brand(
        image_url, spec, fal_client,
    ))

    results = await asyncio.gather(afv_task, brand_task, return_exceptions=True)

    afv = _unwrap_afv(results[0], spec)
    brand = _unwrap_brand(results[1])

    # Weighted score: AFV 0.5 + brand 0.5 (no depth)
    afv_score = afv.score if afv.error is None else 0.0
    brand_score = brand.combined_score if brand.error is None else 0.0
    overall = 0.5 * afv_score + 0.5 * brand_score

    # Hard gates
    all_critical = [*afv.critical_failures, *brand.critical_failures]
    if all_critical:
        overall = min(overall, 0.5)

    grade = compute_grade(overall)
    passed = overall >= 0.7 and not all_critical
    duration_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "product_eval.complete",
        score=overall,
        grade=grade,
        passed=passed,
        afv_score=afv_score,
        brand_score=brand_score,
        duration_ms=duration_ms,
    )

    return {
        "overall": round(overall, 4),
        "grade": grade,
        "passed": passed,
        "afv": afv.model_dump(),
        "brand": brand.model_dump(),
        "depth": None,
        "critical_failures": all_critical,
        "duration_ms": duration_ms,
        "spec_id": spec.shot_id,
        "image_url": image_url,
    }


async def _run_afv(
    image_url: str,
    spec: GoldenSpec,
    api_key: str,
    model_id: str,
) -> AFVReport:
    return await verify_facts(
        image_url, spec.atomic_facts, api_key=api_key, model_id=model_id,
    )


async def _run_brand(
    image_url: str,
    spec: GoldenSpec,
    fal_client: Any,
) -> BrandReport:
    return await evaluate_brand(
        image_url, spec.expected_texts, spec.brand_colors_hex, spec.rois, fal_client,
    )


def _unwrap_afv(result: AFVReport | BaseException, spec: GoldenSpec) -> AFVReport:
    if isinstance(result, BaseException):
        logger.error("product_eval.afv_failed", error=str(result))
        return AFVReport(
            facts=spec.atomic_facts,
            verdicts=[],
            score=0.0,
            category_breakdown={},
            error=str(result),
        )
    return result


def _unwrap_brand(result: BrandReport | BaseException) -> BrandReport:
    if isinstance(result, BaseException):
        logger.error("product_eval.brand_failed", error=str(result))
        from product_fidelity_lab.models.evaluation import ColorScore, TextMatchScore

        return BrandReport(
            text_score=TextMatchScore(matches=[], score=0.0),
            color_score=ColorScore(
                brand_colors_hex=[], extracted_colors_hex=[], pairs=[], score=0.0,
            ),
            combined_score=0.0,
            error=str(result),
        )
    return result
