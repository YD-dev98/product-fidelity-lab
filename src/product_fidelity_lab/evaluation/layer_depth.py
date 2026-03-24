"""Layer 2: Structural Integrity via Depth Anything V2 (supporting signal)."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray
from PIL import Image
from skimage.metrics import structural_similarity  # type: ignore[import-untyped]

from product_fidelity_lab.generation.client import FalClient
from product_fidelity_lab.models.evaluation import DepthScore

if TYPE_CHECKING:
    from product_fidelity_lab.models.golden_spec import ROI

logger = structlog.get_logger()

DEPTH_MODEL = "fal-ai/image-preprocessors/depth-anything/v2"


async def get_depth_map(
    image_url: str,
    fal_client: FalClient,
    cache: object | None = None,
) -> NDArray[np.float64]:
    """Get a depth map from Depth Anything V2 via fal.ai. Returns normalized [0,1] array.

    Uses ResultCache if provided to avoid redundant API calls.
    """
    from typing import Any

    from product_fidelity_lab.evaluation.image_fetch import fetch_image_bytes

    async def _call_depth() -> dict[str, Any]:
        result = await fal_client.subscribe(DEPTH_MODEL, {"image_url": image_url})
        return result

    cache_inputs: dict[str, Any] = {"image_url": image_url, "model": DEPTH_MODEL}
    if cache is not None:
        entry = await cache.get_or_compute(cache_inputs, _call_depth)  # type: ignore[union-attr]
        result: dict[str, Any] = entry["result"]  # type: ignore[reportUnknownVariableType]
    else:
        result = await _call_depth()

    depth_url = str(result["image"]["url"])  # type: ignore[reportUnknownArgumentType]
    depth_bytes = await fetch_image_bytes(depth_url)
    depth_img = Image.open(BytesIO(depth_bytes)).convert("L")
    depth_arr = np.array(depth_img, dtype=np.float64) / 255.0
    return depth_arr


def crop_depth_to_roi(
    depth: NDArray[np.float64],
    roi: ROI,
) -> NDArray[np.float64]:
    """Crop a depth map to a normalized ROI."""
    h, w = depth.shape
    x1 = int(roi.x * w)
    y1 = int(roi.y * h)
    x2 = int((roi.x + roi.width) * w)
    y2 = int((roi.y + roi.height) * h)
    return depth[y1:y2, x1:x2]


def compare_depth(
    golden_depth: NDArray[np.float64],
    generated_depth: NDArray[np.float64],
    roi: ROI | None = None,
) -> DepthScore:
    """Compare two depth maps using SSIM, Pearson correlation, and MSE."""
    if roi is not None:
        golden_depth = crop_depth_to_roi(golden_depth, roi)
        generated_depth = crop_depth_to_roi(generated_depth, roi)

    # Resize to same dimensions
    if golden_depth.shape != generated_depth.shape:
        from PIL import Image as PILImage

        gen_img = PILImage.fromarray((generated_depth * 255).astype(np.uint8))
        gen_img = gen_img.resize((golden_depth.shape[1], golden_depth.shape[0]))
        generated_depth = np.array(gen_img, dtype=np.float64) / 255.0

    # SSIM
    min_dim = min(golden_depth.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)
    ssim_val: float = float(
        structural_similarity(  # type: ignore[reportArgumentType]
            golden_depth, generated_depth, win_size=win_size, data_range=1.0
        )
    )

    # Pearson correlation
    g_flat = golden_depth.flatten()
    d_flat = generated_depth.flatten()
    if np.std(g_flat) == 0 or np.std(d_flat) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(g_flat, d_flat)[0, 1])

    # MSE
    mse = float(np.mean((golden_depth - generated_depth) ** 2))

    combined = 0.5 * ssim_val + 0.3 * correlation + 0.2 * (1.0 - mse)

    return DepthScore(
        ssim=ssim_val,
        correlation=correlation,
        mse=mse,
        combined=combined,
        roi_used=roi is not None,
    )
