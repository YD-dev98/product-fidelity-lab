from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.models.preset import StudioPreset
from product_fidelity_lab.models.product import (
    AngleTag,
    AssetType,
    PackagingType,
    ProductAsset,
    ProductProfile,
    ProductStatus,
)
from product_fidelity_lab.storage.product_store import ProductStore, is_usable_reference


@pytest.fixture
async def store(tmp_path: Path) -> ProductStore:
    s = ProductStore(
        db_path=tmp_path / "test.db",
        products_dir=tmp_path / "products",
    )
    await s.initialize()
    return s


class TestProductCRUD:
    @pytest.mark.asyncio
    async def test_create_and_get(self, store: ProductStore) -> None:
        product = await store.create_product("Test Product")
        assert product.name == "Test Product"
        assert product.status == ProductStatus.DRAFT

        fetched = await store.get_product(product.id)
        assert fetched is not None
        assert fetched.id == product.id
        assert fetched.name == "Test Product"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: ProductStore) -> None:
        assert await store.get_product("doesnotexist") is None

    @pytest.mark.asyncio
    async def test_list_products(self, store: ProductStore) -> None:
        await store.create_product("Product A")
        await store.create_product("Product B")

        products = await store.list_products()
        assert len(products) == 2

    @pytest.mark.asyncio
    async def test_update_product_status(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.update_product(product.id, status=ProductStatus.PROFILED)

        fetched = await store.get_product(product.id)
        assert fetched is not None
        assert fetched.status == ProductStatus.PROFILED

    @pytest.mark.asyncio
    async def test_update_product_profile(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        profile = ProductProfile(
            product_id=product.id,
            brand_texts=["BRAND"],
            brand_colors_hex=["#ff0000"],
            packaging_type=PackagingType.BOTTLE,
            ingest_confidence=0.8,
        )
        await store.update_product(product.id, profile=profile)

        fetched_profile = await store.get_profile(product.id)
        assert fetched_profile is not None
        assert fetched_profile.brand_texts == ["BRAND"]
        assert fetched_profile.packaging_type == PackagingType.BOTTLE
        assert fetched_profile.ingest_confidence == 0.8

    @pytest.mark.asyncio
    async def test_delete_product(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.delete_product(product.id)
        assert await store.get_product(product.id) is None


class TestAssetCRUD:
    @pytest.mark.asyncio
    async def test_add_and_get_assets(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        asset = ProductAsset(
            id="a1",
            product_id=product.id,
            asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/test.jpg",
            width=800,
            height=600,
            angle_tag=AngleTag.FRONT,
        )
        await store.add_asset(asset)

        assets = await store.get_assets(product.id)
        assert len(assets) == 1
        assert assets[0].id == "a1"
        assert assets[0].angle_tag == AngleTag.FRONT

    @pytest.mark.asyncio
    async def test_filter_assets_by_type(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=100, height=100,
        ))
        await store.add_asset(ProductAsset(
            id="a2", product_id=product.id, asset_type=AssetType.ALPHA_MASK,
            file_path="/tmp/2.png", width=100, height=100,
        ))

        raw = await store.get_assets(product.id, asset_type=AssetType.RAW_UPLOAD)
        assert len(raw) == 1
        assert raw[0].id == "a1"

    @pytest.mark.asyncio
    async def test_update_asset_angle(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=100, height=100,
        ))

        await store.update_asset("a1", angle_tag=AngleTag.FRONT_LEFT_45)
        asset = await store.get_asset("a1")
        assert asset is not None
        assert asset.angle_tag == AngleTag.FRONT_LEFT_45


class TestPresetCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get_preset(self, store: ProductStore) -> None:
        preset = StudioPreset(preset_id="test", name="Test Preset")
        await store.save_preset(preset, is_builtin=True)

        fetched = await store.get_preset("test")
        assert fetched is not None
        assert fetched.name == "Test Preset"

    @pytest.mark.asyncio
    async def test_list_presets(self, store: ProductStore) -> None:
        await store.save_preset(
            StudioPreset(preset_id="a", name="A"), is_builtin=True
        )
        await store.save_preset(
            StudioPreset(preset_id="b", name="B"), is_builtin=False
        )

        presets = await store.list_presets()
        assert len(presets) == 2


class TestIsUsableReference:
    def test_raw_upload_with_dimensions(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
        )
        assert is_usable_reference(asset) is True

    def test_alpha_mask_not_usable(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.ALPHA_MASK,
            file_path="/tmp/1.png", width=800, height=600,
        )
        assert is_usable_reference(asset) is False

    def test_no_dimensions_not_usable(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg",
        )
        assert is_usable_reference(asset) is False

    def test_excluded_not_usable(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            metadata={"excluded": True},
        )
        assert is_usable_reference(asset) is False

    def test_required_views_with_unknown_angle(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.UNKNOWN,
        )
        assert is_usable_reference(asset, required_views=[AngleTag.FRONT]) is False

    def test_required_views_with_matching_angle(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.FRONT,
        )
        assert is_usable_reference(asset, required_views=[AngleTag.FRONT]) is True

    def test_required_views_with_wrong_angle(self) -> None:
        asset = ProductAsset(
            id="a1", product_id="p1", asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.BACK,
        )
        assert is_usable_reference(asset, required_views=[AngleTag.FRONT]) is False
