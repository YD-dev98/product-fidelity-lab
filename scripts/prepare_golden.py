"""Upload golden images to fal.ai and pre-compute depth maps.

This populates image_url in each spec and caches golden depth maps.
All operations are free (fal.ai storage + Depth Anything V2).

Usage: uv run python scripts/prepare_golden.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from product_fidelity_lab.config import get_settings

# Ensure fal-client picks up the key
os.environ.setdefault("FAL_KEY", get_settings().fal_key)
import numpy as np

from product_fidelity_lab.evaluation.layer_depth import get_depth_map
from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.storage.fal_storage import FalStorage


async def main() -> None:
    settings = get_settings()
    golden_dir = settings.data_dir / "golden"
    specs_dir = golden_dir / "specs"
    depth_dir = settings.data_dir / "cache" / "golden_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    fal_storage = FalStorage(cache_file=settings.data_dir / "cache" / "upload_cache.json")
    fal_client = FalClient(
        timeout_s=settings.fal_timeout_s,
        max_concurrent=settings.fal_max_concurrent,
    )

    spec_files = sorted(specs_dir.glob("*.json"))
    print(f"Found {len(spec_files)} specs")

    for i, spec_path in enumerate(spec_files, 1):
        spec = json.loads(spec_path.read_text())
        shot_id = spec["shot_id"]
        image_path = golden_dir / spec["image_path"]

        if not image_path.exists():
            print(f"  [{i}/{len(spec_files)}] SKIP {shot_id} (image not found)")
            continue

        # 1. Upload image
        needs_upload = spec.get("image_url") is None
        if needs_upload:
            print(f"  [{i}/{len(spec_files)}] Uploading {shot_id}...", end=" ", flush=True)
            try:
                url = await fal_storage.upload_image(image_path)
                spec["image_url"] = url
                spec_path.write_text(json.dumps(spec, indent=2))
                print(f"OK ({url[:60]}...)")
            except Exception as e:
                print(f"UPLOAD FAILED: {e}")
                continue
        else:
            url = spec["image_url"]
            print(f"  [{i}/{len(spec_files)}] {shot_id} already uploaded")

        # 2. Compute depth map
        depth_path = depth_dir / f"{shot_id}.npy"
        if depth_path.exists():
            print("    Depth map cached")
            continue

        print("    Computing depth map...", end=" ", flush=True)
        try:
            depth = await get_depth_map(url, fal_client)
            np.save(str(depth_path), depth)
            print(f"OK ({depth.shape})")
        except Exception as e:
            print(f"DEPTH FAILED: {e}")

    print("\nDone! Golden images uploaded and depth maps cached.")


if __name__ == "__main__":
    asyncio.run(main())
