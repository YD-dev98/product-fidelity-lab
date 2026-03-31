from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.models.product import (
    AngleTag,
    AssetType,
    PackagingType,
    ProductAsset,
    ProductProfile,
)
from product_fidelity_lab.storage.product_store import ProductStore


@pytest.fixture
async def store(tmp_path: Path) -> ProductStore:
    s = ProductStore(
        db_path=tmp_path / "test.db",
        products_dir=tmp_path / "products",
    )
    await s.initialize()
    return s


class TestProfileConfirm:
    @pytest.mark.asyncio
    async def test_confirm_critical_texts(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        profile = ProductProfile(
            product_id=product.id,
            brand_texts=["BRAND", "EXTRA"],
            ingest_confidence=0.5,
        )
        await store.update_product(product.id, profile=profile)

        # Simulate confirm by updating profile
        fetched = await store.get_profile(product.id)
        assert fetched is not None
        fetched.critical_texts = ["BRAND"]
        await store.update_product(product.id, profile=fetched)

        result = await store.get_profile(product.id)
        assert result is not None
        assert result.critical_texts == ["BRAND"]

    @pytest.mark.asyncio
    async def test_confirm_packaging_type(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        profile = ProductProfile(
            product_id=product.id,
            packaging_type=PackagingType.UNKNOWN,
        )
        await store.update_product(product.id, profile=profile)

        fetched = await store.get_profile(product.id)
        assert fetched is not None
        fetched.packaging_type = PackagingType.BOTTLE
        await store.update_product(product.id, profile=fetched)

        result = await store.get_profile(product.id)
        assert result is not None
        assert result.packaging_type == PackagingType.BOTTLE

    @pytest.mark.asyncio
    async def test_angle_override_updates_asset(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.UNKNOWN,
        ))

        # Override angle
        await store.update_asset("a1", angle_tag=AngleTag.FRONT)

        asset = await store.get_asset("a1")
        assert asset is not None
        assert asset.angle_tag == AngleTag.FRONT

    @pytest.mark.asyncio
    async def test_angle_override_updates_profile_views(
        self, store: ProductStore
    ) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.UNKNOWN,
        ))

        profile = ProductProfile(product_id=product.id, views={})
        # Apply angle override to profile
        profile.views[AngleTag.FRONT] = "a1"
        await store.update_product(product.id, profile=profile)

        fetched = await store.get_profile(product.id)
        assert fetched is not None
        assert fetched.views[AngleTag.FRONT] == "a1"
