"""Layer 1: Atomic Fact Verification via Gemini 2.5 Pro."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from google import genai

from product_fidelity_lab.models.evaluation import (
    IMPORTANCE_WEIGHTS,
    AFVReport,
    FactVerdict,
)
from product_fidelity_lab.models.golden_spec import AtomicFact

logger = structlog.get_logger()

VERIFY_PROMPT = """You are a strict image evaluator. Examine this image carefully.

For each numbered statement below, determine if it is TRUE or FALSE in the image.
Be strict — if you cannot clearly confirm a statement, mark it FALSE.

Statements:
{facts_block}

Respond with a JSON array of objects, one per statement, in the same order:
[
  {{"id": "F1", "verdict": true, "confidence": 0.95, "reasoning": "..."}},
  ...
]

Only output the JSON array, no other text."""

BOOTSTRAP_PROMPT = """You are an expert product photographer analyzing a reference photo.

Generate a comprehensive list of atomic facts about this image. Each fact should be
a single, verifiable statement about one visual aspect.

Categorize each fact as: GEOMETRY, MATERIAL, LIGHTING, COLOR, DETAIL, TEXT, or CONTEXT.

Also identify regions of interest:
- The bounding box (normalized 0-1, as x, y, width, height) for the product, label, and logo.

Respond with JSON:
{{
  "facts": [
    {{"id": "F1", "category": "GEOMETRY", "fact": "...", "importance": "high"}},
    ...
  ],
  "rois": [
    {{"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.6, "label": "product"}},
    ...
  ]
}}

Only output JSON."""


async def verify_facts(
    generated_image_url: str,
    facts: list[AtomicFact],
    *,
    api_key: str,
    model_id: str,
    cache: Any | None = None,
) -> AFVReport:
    """Verify atomic facts against a generated image using Gemini.

    Sends all facts in a single call. Uses ResultCache if provided.
    """
    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    facts_block = "\n".join(f"{f.id}: {f.fact}" for f in facts)
    prompt = VERIFY_PROMPT.format(facts_block=facts_block)

    image_bytes = await fetch_image_bytes(generated_image_url)

    async def _call_gemini() -> dict[str, Any]:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[
                genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return {"text": response.text or ""}

    cache_inputs = {
        "image_url": generated_image_url,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": model_id,
    }
    if cache is not None:
        entry = await cache.get_or_compute(cache_inputs, _call_gemini)
        response_text: str = entry["result"]["text"]
    else:
        raw = await _call_gemini()
        response_text = raw["text"]

    verdicts = _parse_verdicts(response_text, facts)

    # Scoring
    total_weight = 0.0
    weighted_sum = 0.0
    category_scores: dict[str, list[float]] = {}
    critical_failures: list[str] = []

    for fact, verdict in zip(facts, verdicts, strict=True):
        w = IMPORTANCE_WEIGHTS.get(fact.importance, 1.0)
        total_weight += w
        weighted_sum += w * (1.0 if verdict.verdict else 0.0)

        cat = fact.category
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(1.0 if verdict.verdict else 0.0)

        if fact.critical and not verdict.verdict:
            critical_failures.append(f"Critical fact failed: {fact.fact}")

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    category_breakdown = {
        cat: sum(vals) / len(vals) for cat, vals in category_scores.items()
    }

    return AFVReport(
        facts=facts,
        verdicts=verdicts,
        score=score,
        category_breakdown=category_breakdown,
        critical_failures=critical_failures,
    )


def _parse_verdicts(
    text: str,
    facts: list[AtomicFact],
) -> list[FactVerdict]:
    """Parse Gemini response into FactVerdicts."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data: list[dict[str, Any]] = json.loads(cleaned)
        verdicts: list[FactVerdict] = []
        for item in data:
            verdicts.append(
                FactVerdict(
                    fact_id=str(item.get("id", "")),
                    verdict=bool(item.get("verdict", False)),
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=str(item.get("reasoning", "")),
                )
            )
        return verdicts
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("afv.parse_failed", text_length=len(text))
        msg = (
            f"AFV parse failure — could not extract verdicts "
            f"from model response ({len(text)} chars)"
        )
        raise RuntimeError(msg) from exc


async def bootstrap_facts(
    image_url: str,
    *,
    api_key: str,
    model_id: str = "gemini-2.5-pro",
) -> dict[str, Any]:
    """Generate draft atomic facts and ROIs from a golden image. Dev tool only."""
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[
            genai.types.Part.from_uri(file_uri=image_url, mime_type="image/jpeg"),
            BOOTSTRAP_PROMPT,
        ],
    )

    text = (response.text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return json.loads(text)  # type: ignore[no-any-return]
