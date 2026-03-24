"""Layer 3: Brand Integrity — OCR text matching + color fidelity."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from io import BytesIO
from typing import cast

import numpy as np
import structlog
from numpy.typing import NDArray
from PIL import Image

from product_fidelity_lab.evaluation.color import (
    compare_to_brand_colors,
    crop_to_roi,
    extract_dominant_colors,
)
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.evaluation import (
    BrandReport,
    ColorPair,
    ColorScore,
    TextMatch,
    TextMatchScore,
)
from product_fidelity_lab.models.golden_spec import ROI, ExpectedText

logger = structlog.get_logger()

OCR_MODEL = "fal-ai/got-ocr/v2"


async def extract_text(
    image_url: str,
    fal_client: FalClient,
    roi: ROI | None = None,
    cache: object | None = None,
) -> list[str]:
    """Extract text from an image using GOT-OCR 2.0 via fal.ai.

    If ROI is provided, the image is cropped before OCR.
    Uses ResultCache if provided.
    """
    # If ROI, download via validated path + crop + re-upload
    ocr_url = image_url
    if roi is not None:
        import tempfile
        from pathlib import Path as _Path

        from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

        img_bytes = await fetch_image_bytes(image_url)
        img = Image.open(BytesIO(img_bytes))
        cropped = crop_to_roi(img, roi)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cropped.save(tmp, format="PNG")
            tmp_path = tmp.name

        import fal_client as fal_mod

        ocr_url = await fal_mod.upload_file_async(_Path(tmp_path))  # type: ignore[reportUnknownMemberType]

    async def _call_ocr() -> dict[str, object]:
        return await fal_client.subscribe(
            OCR_MODEL, {"input_image_urls": [ocr_url]},
        )

    from typing import Any

    cache_inputs: dict[str, Any] = {
        "image_url": image_url, "model": OCR_MODEL,
        "roi": [roi.x, roi.y, roi.width, roi.height] if roi else None,
    }
    if cache is not None:
        entry = await cache.get_or_compute(cache_inputs, _call_ocr)  # type: ignore[union-attr]
        result = dict(entry["result"])  # type: ignore[reportUnknownArgumentType]
    else:
        result = dict(await _call_ocr())

    # GOT-OCR v2 returns {"outputs": ["extracted text", ...]}
    texts: list[str] = []
    outputs: object = result.get("outputs", [])  # type: ignore[reportUnknownMemberType]
    if isinstance(outputs, list):
        for chunk in cast(list[object], outputs):
            if isinstance(chunk, str) and chunk.strip():
                texts.extend(
                    line.strip() for line in chunk.split("\n") if line.strip()
                )

    logger.info("ocr.extracted", text_count=len(texts))
    return texts


def _tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for matching."""
    return re.findall(r"\b\w+\b", text.lower())


def compare_text(
    expected_tokens: list[ExpectedText],
    extracted_texts: list[str],
) -> TextMatchScore:
    """Compare expected text tokens against OCR-extracted text.

    For exact_token mode: checks if expected token sequence appears
    as a contiguous subsequence in the unified OCR token stream.
    For fuzzy mode: uses SequenceMatcher best-match ratio.
    """
    if not expected_tokens:
        return TextMatchScore(
            matches=[], score=1.0, extracted_texts=extracted_texts
        )

    # Build unified token stream from all extracted text
    unified = " ".join(extracted_texts)
    unified_tokens = _tokenize(unified)

    matches: list[TextMatch] = []
    critical_failures: list[str] = []

    for et in expected_tokens:
        if et.match_mode == "exact_token":
            expected_toks = _tokenize(et.text)
            matched = _is_contiguous_subsequence(expected_toks, unified_tokens)
            score = 1.0 if matched else 0.0
        else:
            # Fuzzy mode
            ratio = SequenceMatcher(
                None, et.text.lower(), unified.lower()
            ).ratio()
            matched = ratio >= 0.6
            score = ratio

        matches.append(
            TextMatch(
                expected=et.text,
                matched=matched,
                score=score,
                match_mode=et.match_mode,
                critical=et.critical,
            )
        )

        if et.critical and not matched:
            critical_failures.append(f"Brand text missing: \"{et.text}\"")

    total_score = sum(m.score for m in matches) / len(matches) if matches else 1.0

    return TextMatchScore(
        matches=matches,
        score=total_score,
        critical_failures=critical_failures,
        extracted_texts=extracted_texts,
    )


def _is_contiguous_subsequence(
    needle: list[str],
    haystack: list[str],
) -> bool:
    """Check if needle appears as a contiguous subsequence in haystack."""
    if not needle:
        return True
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


async def evaluate_brand(
    generated_image_url: str,
    expected_texts: list[ExpectedText],
    brand_colors_hex: list[str],
    rois: list[ROI],
    fal_client: FalClient,
    cache: object | None = None,
) -> BrandReport:
    """Run full brand integrity evaluation (OCR + Color)."""
    # Find relevant ROIs
    text_roi = next((r for r in rois if r.label in ("label", "product")), None)
    color_roi = next((r for r in rois if r.label == "product"), None)

    # OCR — skip if no expected texts, propagate provider failures
    if expected_texts:
        extracted = await extract_text(
            generated_image_url, fal_client, roi=text_roi, cache=cache,
        )
    else:
        extracted = []
    text_result = compare_text(expected_texts, extracted)

    # Color
    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    img_bytes = await fetch_image_bytes(generated_image_url)
    img = Image.open(BytesIO(img_bytes))

    extracted_lab = extract_dominant_colors(img, roi=color_roi, n_colors=5)
    color_score_val, color_pairs = compare_to_brand_colors(
        extracted_lab, brand_colors_hex
    )

    # Convert extracted LAB back to hex for reporting
    from skimage.color import lab2rgb  # type: ignore[import-untyped]

    from product_fidelity_lab.evaluation.color import rgb_to_hex

    extracted_hex: list[str] = []
    for lab in extracted_lab:
        lab_3d = lab.reshape(1, 1, 3)
        rgb_arr = np.asarray(lab2rgb(lab_3d), dtype=np.float64)
        rgb: NDArray[np.float64] = rgb_arr.reshape(3) * 255.0
        extracted_hex.append(rgb_to_hex(rgb))

    color_result = ColorScore(
        brand_colors_hex=brand_colors_hex,
        extracted_colors_hex=extracted_hex,
        pairs=[
            ColorPair(brand_hex=p[0], closest_extracted_hex=p[1], delta_e=p[2])
            for p in color_pairs
        ],
        score=color_score_val,
    )

    # When no expected texts exist, use color-only instead of granting a free 1.0 text score
    if expected_texts:
        combined = 0.4 * text_result.score + 0.6 * color_result.score
    else:
        combined = color_result.score

    all_critical: list[str] = [*text_result.critical_failures]

    return BrandReport(
        text_score=text_result,
        color_score=color_result,
        combined_score=combined,
        critical_failures=all_critical,
    )
