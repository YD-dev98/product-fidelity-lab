"""Color extraction and Delta-E CIE2000 comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from product_fidelity_lab.models.golden_spec import ROI


def hex_to_rgb(hex_color: str) -> NDArray[np.float64]:
    """Convert '#RRGGBB' to [R, G, B] float array (0-255)."""
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def rgb_to_hex(rgb: NDArray[np.float64]) -> str:
    """Convert [R, G, B] (0-255) to '#RRGGBB'."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def crop_to_roi(img: Image.Image, roi: ROI) -> Image.Image:
    """Crop a PIL image to the normalized ROI coordinates."""
    w, h = img.size
    left = int(roi.x * w)
    top = int(roi.y * h)
    right = int((roi.x + roi.width) * w)
    bottom = int((roi.y + roi.height) * h)
    return img.crop((left, top, right, bottom))


def extract_dominant_colors(
    image: Image.Image,
    roi: ROI | None = None,
    n_colors: int = 5,
) -> list[NDArray[np.float64]]:
    """Extract dominant colors as LAB arrays via KMeans clustering.

    Args:
        image: PIL Image to analyze.
        roi: Optional ROI to crop to before analysis.
        n_colors: Number of dominant colors to extract.

    Returns:
        List of LAB color arrays.
    """
    if roi is not None:
        image = crop_to_roi(image, roi)

    img_rgb = np.array(image.convert("RGB"))
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)

    # Subsample for performance if image is large
    if len(pixels) > 10000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]

    n_colors = min(n_colors, len(pixels))
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)  # type: ignore[reportArgumentType]
    kmeans.fit(pixels)  # type: ignore[reportUnknownMemberType]

    centers_rgb: NDArray[np.float64] = np.asarray(
        kmeans.cluster_centers_, dtype=np.float64,  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    ) / 255.0
    # rgb2lab expects shape (1, N, 3) for batch conversion
    lab_raw: NDArray[np.float64] = rgb2lab(centers_rgb.reshape(1, -1, 3))  # type: ignore[reportUnknownVariableType]
    centers_lab: NDArray[np.float64] = np.asarray(
        lab_raw, dtype=np.float64,
    ).reshape(-1, 3)
    return [centers_lab[i] for i in range(len(centers_lab))]


def delta_e_cie2000(
    lab1: NDArray[np.float64],
    lab2: NDArray[np.float64],
) -> float:
    """Compute Delta-E CIE2000 between two LAB colors."""
    result: float = float(
        deltaE_ciede2000(lab1.reshape(1, 1, 3), lab2.reshape(1, 1, 3))[0, 0]
    )
    return result


def compare_to_brand_colors(
    extracted_lab: list[NDArray[np.float64]],
    brand_hex: list[str],
) -> tuple[float, list[tuple[str, str, float]]]:
    """Compare extracted colors to brand palette.

    Returns:
        Tuple of (score, list of (brand_hex, closest_extracted_hex, delta_e)).
    """
    if not brand_hex or not extracted_lab:
        return 1.0, []

    # Convert brand colors to LAB
    brand_rgb = [hex_to_rgb(h) / 255.0 for h in brand_hex]
    brand_lab: list[NDArray[np.float64]] = [
        np.asarray(rgb2lab(rgb.reshape(1, 1, 3)), dtype=np.float64).reshape(3) for rgb in brand_rgb
    ]

    # For each brand color, find closest extracted color
    pairs: list[tuple[str, str, float]] = []
    delta_es: list[float] = []

    for i, b_lab in enumerate(brand_lab):
        min_de = float("inf")
        closest_idx = 0
        for j, e_lab in enumerate(extracted_lab):
            de = delta_e_cie2000(b_lab, e_lab)
            if de < min_de:
                min_de = de
                closest_idx = j

        # Convert closest extracted LAB back to hex for reporting
        e_rgb = extracted_lab[closest_idx].copy()
        # LAB -> RGB: need to go through skimage
        from skimage.color import lab2rgb  # type: ignore[import-untyped]

        lab_3d = e_rgb.reshape(1, 1, 3)
        e_rgb_arr: NDArray[np.float64] = np.asarray(
            lab2rgb(lab_3d), dtype=np.float64,
        ).reshape(3) * 255.0
        e_hex = rgb_to_hex(e_rgb_arr)

        pairs.append((brand_hex[i], e_hex, min_de))
        delta_es.append(min_de)

    avg_de = float(np.mean(delta_es))
    score = max(0.0, 1.0 - avg_de / 50.0)
    return score, pairs
