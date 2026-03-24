"""Bootstrap golden specs for all training + validation images.

Sends each image to Gemini 2.5 Pro to generate draft atomic facts, ROIs,
expected text, and brand color suggestions. Saves draft specs to data/golden/specs/.

Usage: uv run python scripts/bootstrap_specs.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from google import genai

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from product_fidelity_lab.config import get_settings

BOOTSTRAP_PROMPT = """You are an expert product photographer analyzing a reference photo for an AI evaluation system.

Analyze this image and produce a comprehensive specification. Be thorough and precise.

1. **Atomic Facts**: Generate 8-15 verifiable statements about the image. Each fact should be a single, specific, verifiable claim. Categorize each as:
   - GEOMETRY: shape, position, angle, composition
   - MATERIAL: surface texture, finish, reflections
   - LIGHTING: light direction, shadows, highlights, exposure
   - COLOR: specific colors present, gradients, contrasts
   - DETAIL: fine details, engravings, patterns, sharpness
   - TEXT: any visible text, labels, logos
   - CONTEXT: background, surface, environment

   Mark importance as "high" (essential to the shot), "medium" (noticeable), or "low" (subtle detail).
   Mark critical=true for facts that are absolutely essential — if wrong, the image has fundamentally failed.

2. **ROIs** (Regions of Interest): Identify bounding boxes (normalized 0-1 coordinates: x, y, width, height) for:
   - "product": the main product
   - "label": any text/label area on the product
   - "logo": any logo area (if visible)

3. **Expected Text**: List any text visible on the product (brand name, description, etc).
   Mark critical=true for the brand name.

4. **Brand Colors**: List the dominant product/brand colors as hex codes.

5. **Shot Type**: Classify as one of: "hero", "orbit", "vertical", "detail", "challenge"

6. **Description**: One sentence describing this shot.

Respond with ONLY this JSON structure:
{
  "atomic_facts": [
    {"id": "F1", "category": "GEOMETRY", "fact": "...", "critical": false, "importance": "high"},
    ...
  ],
  "rois": [
    {"x": 0.1, "y": 0.1, "width": 0.6, "height": 0.8, "label": "product"},
    ...
  ],
  "expected_texts": [
    {"text": "Brand Name", "critical": true, "match_mode": "exact_token"},
    ...
  ],
  "brand_colors_hex": ["#c8a951", "#1a1a1a"],
  "shot_type": "hero",
  "description": "Front-left hero shot of the bottle on white surface"
}"""


def image_to_shot_id(filename: str) -> str:
    """Convert filename to a shot_id."""
    return Path(filename).stem


async def bootstrap_one(
    client: genai.Client,
    image_path: Path,
    category: str,
) -> dict[str, object]:
    """Bootstrap a single image's spec via Gemini."""
    image_bytes = image_path.read_bytes()

    # Determine mime type
    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    # Use Flash for bootstrapping (generous free tier), Pro for evaluation
    model = "gemini-2.5-flash"

    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            BOOTSTRAP_PROMPT,
        ],
    )

    text = (response.text or "").strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data: dict[str, object] = json.loads(text)

    shot_id = image_to_shot_id(image_path.name)

    # Build the full spec
    spec: dict[str, object] = {
        "shot_id": shot_id,
        "image_path": f"{category}/{image_path.name}",
        "image_url": None,
        "category": category,
        "shot_type": data.get("shot_type", "hero"),
        "atomic_facts": data.get("atomic_facts", []),
        "expected_texts": data.get("expected_texts", []),
        "brand_colors_hex": data.get("brand_colors_hex", []),
        "rois": data.get("rois", []),
        "spec_version": 1,
        "description": data.get("description", f"{shot_id} shot"),
        "challenge_type": None,
    }

    return spec


async def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    golden_dir = settings.data_dir / "golden"
    specs_dir = golden_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    images: list[tuple[Path, str]] = []
    for img in sorted((golden_dir / "training").glob("*")):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            images.append((img, "training"))
    for img in sorted((golden_dir / "validation").glob("*")):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            images.append((img, "validation"))

    print(f"Found {len(images)} images to bootstrap")

    for i, (img_path, category) in enumerate(images, 1):
        shot_id = image_to_shot_id(img_path.name)
        spec_path = specs_dir / f"{shot_id}.json"

        if spec_path.exists():
            print(f"  [{i}/{len(images)}] SKIP {shot_id} (spec exists)")
            continue

        print(f"  [{i}/{len(images)}] Bootstrapping {shot_id}...", end=" ", flush=True)
        try:
            spec = await bootstrap_one(client, img_path, category)
            spec_path.write_text(json.dumps(spec, indent=2))
            n_facts = len(spec.get("atomic_facts", []))  # type: ignore[arg-type]
            print(f"OK ({n_facts} facts)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone! Specs saved to {specs_dir}/")
    print("Review and curate each spec before freezing.")


if __name__ == "__main__":
    asyncio.run(main())
