"""Build a ProductProfile from ingested assets."""

from __future__ import annotations

from product_fidelity_lab.models.product import (
    AngleTag,
    AssetType,
    PackagingType,
    ProductAsset,
    ProductProfile,
)


class ProductProfileBuilder:
    """Assembles a ProductProfile from ingested assets."""

    def __init__(self, product_id: str, assets: list[ProductAsset]) -> None:
        self._product_id = product_id
        self._assets = assets
        self._brand_texts: list[str] = []
        self._critical_texts: list[str] = []
        self._colors_hex: list[str] = []
        self._logo_asset_id: str | None = None
        self._label_asset_id: str | None = None
        self._alpha_mask_asset_id: str | None = None
        self._depth_summary: str | None = None
        self._material_summary: str | None = None
        self._packaging_type = PackagingType.UNKNOWN
        self._confidence_signals: list[bool] = []

    def set_brand_texts(self, texts: list[str]) -> ProductProfileBuilder:
        self._brand_texts = texts
        self._confidence_signals.append(bool(texts))
        return self

    def set_colors(self, colors_hex: list[str]) -> ProductProfileBuilder:
        self._colors_hex = colors_hex
        self._confidence_signals.append(bool(colors_hex))
        return self

    def set_logo(self, asset_id: str) -> ProductProfileBuilder:
        self._logo_asset_id = asset_id
        self._confidence_signals.append(True)
        return self

    def set_label(self, asset_id: str) -> ProductProfileBuilder:
        self._label_asset_id = asset_id
        self._confidence_signals.append(True)
        return self

    def set_alpha_mask(self, asset_id: str) -> ProductProfileBuilder:
        self._alpha_mask_asset_id = asset_id
        self._confidence_signals.append(True)
        return self

    def set_summaries(
        self,
        *,
        depth: str | None = None,
        material: str | None = None,
        packaging: PackagingType | None = None,
    ) -> ProductProfileBuilder:
        self._depth_summary = depth
        self._material_summary = material
        if packaging is not None:
            self._packaging_type = packaging
        return self

    def build(self) -> ProductProfile:
        # Build views mapping from angle-tagged visual assets
        views: dict[AngleTag, str] = {}
        for asset in self._assets:
            if (
                asset.asset_type in (AssetType.RAW_UPLOAD, AssetType.CLEANED)
                and asset.angle_tag != AngleTag.UNKNOWN
            ):
                views[asset.angle_tag] = asset.id

        # Compute ingest confidence: ratio of successful enrichments
        if self._confidence_signals:
            confidence = sum(self._confidence_signals) / len(self._confidence_signals)
        else:
            confidence = 0.0

        # Having angle-tagged views boosts confidence
        if views:
            confidence = min(1.0, confidence + 0.1)

        return ProductProfile(
            product_id=self._product_id,
            brand_texts=self._brand_texts,
            critical_texts=self._critical_texts,
            brand_colors_hex=self._colors_hex,
            logo_asset_id=self._logo_asset_id,
            label_asset_id=self._label_asset_id,
            alpha_mask_asset_id=self._alpha_mask_asset_id,
            views=views,
            depth_summary=self._depth_summary,
            material_summary=self._material_summary,
            packaging_type=self._packaging_type,
            ingest_confidence=round(confidence, 2),
        )
