"""Two-stage candidate ranking: sanity filters then Gemini scoring.

Stage 1 (sanity_filter): cheap pass/fail checks on every candidate.
Stage 2 (gemini_rank): vision scoring on the 3-4 survivors.

Fallback rule: if Stage 1 filters everything, keep the least-bad candidate
with filter_passed=False and a warning, rather than failing the run.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import numpy as np
import structlog
from PIL import Image

from product_fidelity_lab.models.preset import Candidate, StudioPreset
from product_fidelity_lab.models.product import ProductProfile

logger = structlog.get_logger()

RANKER_VERSION = "v1"

# ── Stage 1: Sanity Filters ─────────────────────────────────────────


async def sanity_filter(
    candidates: list[Candidate],
    profile: ProductProfile,
    preset: StudioPreset,
    *,
    fal_client: Any | None = None,
) -> list[Candidate]:
    """Run cheap pass/fail checks on every candidate.

    Mutates each candidate's filter_passed and filter_reasons in place.
    Returns all candidates (both passed and failed).

    If fal_client is provided and the preset is text-critical, runs an
    OCR absence check. Otherwise that check is skipped.

    If all fail, the one with the fewest reasons is marked as the fallback
    (filter_passed stays False but it remains selectable).
    """
    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    check_ocr = (
        preset.supports_text_critical
        and profile.critical_texts
        and fal_client is not None
    )

    for candidate in candidates:
        reasons: list[str] = []

        # 1. Fetch image and check for blank/corrupt
        try:
            img_bytes = await fetch_image_bytes(candidate.image_url)
            img = Image.open(BytesIO(img_bytes))
            img.load()
        except Exception:
            reasons.append("corrupt_or_unreachable")
            candidate.filter_passed = False
            candidate.filter_reasons = reasons
            continue

        # 2. Blank/near-blank detection (very low variance = solid color)
        try:
            arr = np.array(img.convert("RGB"), dtype=np.float64)
            pixel_std = float(np.std(arr))
            if pixel_std < 5.0:
                reasons.append("blank_output")
        except Exception:
            reasons.append("image_analysis_error")

        # 3. Subject occupancy — product should take up meaningful area
        try:
            gray = np.array(img.convert("L"), dtype=np.float64)
            h, w = gray.shape
            center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            center_std = float(np.std(center))
            if center_std < 8.0:
                reasons.append("no_subject_detected")
        except Exception:
            pass  # non-critical

        # 4. Color palette sanity
        if profile.brand_colors_hex and not reasons:
            try:
                palette_ok = _check_color_presence(img, profile.brand_colors_hex)
                if not palette_ok:
                    reasons.append("color_palette_mismatch")
            except Exception:
                pass  # non-critical

        # 5. OCR absence on text-critical presets
        if check_ocr and not reasons:
            try:
                from product_fidelity_lab.evaluation.layer_brand import extract_text

                texts = await extract_text(
                    candidate.image_url, fal_client,
                )
                found_any = _check_critical_text_present(
                    profile.critical_texts, texts,
                )
                if not found_any:
                    reasons.append("critical_text_absent")
            except Exception:
                pass  # OCR failure is non-fatal in Stage 1

        candidate.filter_passed = len(reasons) == 0
        candidate.filter_reasons = reasons

    passed = [c for c in candidates if c.filter_passed]
    failed = [c for c in candidates if not c.filter_passed]

    logger.info(
        "ranker.stage1_complete",
        total=len(candidates),
        passed=len(passed),
        failed=len(failed),
    )

    # Fallback: if all filtered, keep the least-bad one
    if not passed and failed:
        least_bad = min(failed, key=lambda c: len(c.filter_reasons))
        logger.warning(
            "ranker.all_filtered_fallback",
            candidate_id=least_bad.candidate_id,
            reasons=least_bad.filter_reasons,
        )
        # Don't flip filter_passed — it stays False so consumers know

    return candidates


def _check_color_presence(
    img: Image.Image,
    brand_colors_hex: list[str],
    threshold_de: float = 35.0,
) -> bool:
    """Check if any brand color is roughly present in the image.

    Uses a loose Delta-E threshold — this is a sanity check, not a fidelity test.
    """
    from skimage.color import rgb2lab  # type: ignore[import-untyped]

    from product_fidelity_lab.evaluation.color import (
        delta_e_cie2000,
        extract_dominant_colors,
        hex_to_rgb,
    )

    if not brand_colors_hex:
        return True  # nothing to check

    extracted_lab = extract_dominant_colors(img, n_colors=5)
    if not extracted_lab:
        return True  # can't check, let it pass

    brand_rgb = [hex_to_rgb(h) / 255.0 for h in brand_colors_hex]
    brand_lab = [
        np.asarray(rgb2lab(rgb.reshape(1, 1, 3)), dtype=np.float64).reshape(3)
        for rgb in brand_rgb
    ]

    # At least one brand color should have a close match in the image
    for b_lab in brand_lab:
        for e_lab in extracted_lab:
            if delta_e_cie2000(b_lab, e_lab) < threshold_de:
                return True

    return False


def _check_critical_text_present(
    critical_texts: list[str],
    extracted_texts: list[str],
) -> bool:
    """Check if at least one critical text token appears in OCR output.

    Uses case-insensitive substring matching — this is a sanity gate,
    not a fidelity test. The full text scoring is done by the evaluator.
    """
    if not critical_texts:
        return True
    unified = " ".join(extracted_texts).lower()
    return any(ct.lower() in unified for ct in critical_texts)


# ── Stage 2: Gemini Scoring ──────────────────────────────────────────

JUDGE_PROMPT = """You are scoring AI-generated product images for quality.

The reference images below show the real product. Compare each candidate against them.

Product context:
- Packaging type: {packaging_type}
- Brand texts that should be visible: {critical_texts}

Requested style:
- Background: {background}
- Lighting: {lighting}
- Camera angle: {camera_angle}
- Render prompt: {render_prompt}

Score each candidate on 4 dimensions (0.0 to 1.0):
- identity: Does this look like the same product as the references?
- label: Are brand text and labels readable and correct?
- style: Does the image match the requested style/lighting/composition above?
- composition: Is the framing, crop, and overall composition professional?

{image_list}

Respond with a JSON array, one object per candidate, in order:
[
  {{"image": 1, "identity": 0.85, "label": 0.7, "style": 0.9, "composition": 0.8}},
  ...
]

Only output the JSON array, no other text."""

# Weights for composite rank_score
SCORE_WEIGHTS = {
    "identity": 0.4,
    "label": 0.3,
    "style": 0.2,
    "composition": 0.1,
}

MAX_JUDGE_CANDIDATES = 4


async def gemini_rank(
    candidates: list[Candidate],
    profile: ProductProfile,
    preset: StudioPreset,
    *,
    reference_urls: list[str],
    render_prompt: str = "",
    api_key: str,
    model_id: str,
) -> list[Candidate]:
    """Score candidates via Gemini vision. Mutates rank scores in place.

    Sends reference images for identity grounding, plus preset style
    details so Gemini can judge style adherence.

    Only scores up to MAX_JUDGE_CANDIDATES. Candidates are sorted
    by rank_score descending after scoring.
    """
    from google import genai

    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes
    from product_fidelity_lab.product.ingest import _detect_mime

    to_score = candidates[:MAX_JUDGE_CANDIDATES]
    if not to_score:
        return candidates

    # Build prompt context
    critical_texts = (
        ", ".join(f'"{t}"' for t in profile.critical_texts)
        or "none specified"
    )
    image_list_parts: list[str] = []
    for i, c in enumerate(to_score, 1):
        image_list_parts.append(f"Candidate {i} (seed={c.seed}):")

    prompt = JUDGE_PROMPT.format(
        packaging_type=profile.packaging_type.value,
        critical_texts=critical_texts,
        background=preset.background,
        lighting=preset.lighting,
        camera_angle=preset.camera_angle.value.replace("_", " "),
        render_prompt=render_prompt or "(no additional style prompt)",
        image_list="\n".join(image_list_parts),
    )

    # Build multimodal content
    contents: list[Any] = []

    # First: reference images for identity grounding
    for i, url in enumerate(reference_urls[:3], 1):
        try:
            ref_bytes = await fetch_image_bytes(url)
            mime = _detect_mime(url)
            contents.append(f"Reference {i}:")
            contents.append(
                genai.types.Part.from_bytes(
                    data=ref_bytes, mime_type=mime,
                )
            )
        except Exception:
            logger.warning("ranker.ref_fetch_failed", url=url)

    # Then: candidate images to score
    for i, c in enumerate(to_score, 1):
        try:
            img_bytes = await fetch_image_bytes(c.image_url)
            mime = _detect_mime(c.image_url)
            contents.append(f"Candidate {i}:")
            contents.append(
                genai.types.Part.from_bytes(
                    data=img_bytes, mime_type=mime,
                )
            )
        except Exception:
            logger.warning("ranker.fetch_failed", candidate_id=c.candidate_id)
            contents.append(f"Candidate {i}: [failed to load]")

    contents.append(prompt)

    try:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=contents,
        )

        text = (response.text or "").strip()
        scores = _parse_judge_response(text, len(to_score))

        for i, score_dict in enumerate(scores):
            if i >= len(to_score):
                break
            c = to_score[i]
            c.identity_score = score_dict.get("identity")
            c.label_score = score_dict.get("label")
            c.style_score = score_dict.get("style")
            c.composition_score = score_dict.get("composition")
            c.rank_score = _compute_rank_score(score_dict)

        logger.info("ranker.stage2_complete", scored=len(scores))

    except Exception:
        logger.warning("ranker.gemini_scoring_failed", count=len(to_score))
        # On failure, leave rank_score as None — candidates are still usable

    # Sort all scored candidates by rank_score (None sorts last)
    candidates.sort(key=lambda c: c.rank_score if c.rank_score is not None else -1.0, reverse=True)
    return candidates


def _parse_judge_response(text: str, expected_count: int) -> list[dict[str, float]]:
    """Parse Gemini JSON response into score dicts."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data: list[dict[str, Any]] = json.loads(cleaned)
        results: list[dict[str, float]] = []
        for item in data:
            results.append({
                "identity": _clamp(item.get("identity", 0.5)),
                "label": _clamp(item.get("label", 0.5)),
                "style": _clamp(item.get("style", 0.5)),
                "composition": _clamp(item.get("composition", 0.5)),
            })
        return results
    except (json.JSONDecodeError, TypeError):
        logger.warning("ranker.parse_failed", text_length=len(text))
        return [{"identity": 0.5, "label": 0.5, "style": 0.5, "composition": 0.5}] * expected_count


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _compute_rank_score(scores: dict[str, float]) -> float:
    total = sum(
        scores.get(dim, 0.5) * weight
        for dim, weight in SCORE_WEIGHTS.items()
    )
    return round(total, 4)


def build_judge_summary(
    candidates: list[Candidate],
    model_id: str,
) -> dict[str, object]:
    """Build a summary of Stage 2 scoring for the trace."""
    scored = [c for c in candidates if c.rank_score is not None]
    top_score = max((c.rank_score for c in scored), default=None)
    return {
        "model": model_id,
        "scored": len(scored),
        "top_score": top_score,
    }


# ── Helpers ──────────────────────────────────────────────────────────


def build_filter_summary(candidates: list[Candidate]) -> dict[str, object]:
    """Build a summary of Stage 1 filter results for the trace."""
    passed = sum(1 for c in candidates if c.filter_passed)
    failed = len(candidates) - passed

    reason_counts: dict[str, int] = {}
    for c in candidates:
        for r in c.filter_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "total": len(candidates),
        "passed": passed,
        "failed": failed,
        "reasons": reason_counts,
    }
