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
)
from product_fidelity_lab.product.validation import validate_compatibility
from product_fidelity_lab.storage.product_store import ProductStore


def _make_profile(**kwargs: object) -> ProductProfile:
    defaults = {
        "product_id": "test",
        "packaging_type": PackagingType.BOTTLE,
        "ingest_confidence": 0.8,
    }
    defaults.update(kwargs)
    return ProductProfile(**defaults)  # type: ignore[arg-type]


def _make_preset(**kwargs: object) -> StudioPreset:
    defaults = {
        "preset_id": "test",
        "name": "Test",
    }
    defaults.update(kwargs)
    return StudioPreset(**defaults)  # type: ignore[arg-type]


@pytest.fixture
async def store(tmp_path: Path) -> ProductStore:
    s = ProductStore(
        db_path=tmp_path / "test.db",
        products_dir=tmp_path / "products",
    )
    await s.initialize()
    return s


async def _add_usable_asset(store: ProductStore, product_id: str, asset_id: str = "a1") -> None:
    """Helper to add a minimal usable asset so min_reference_count passes."""
    await store.add_asset(ProductAsset(
        id=asset_id, product_id=product_id, asset_type=AssetType.RAW_UPLOAD,
        file_path="/tmp/test.jpg", width=800, height=600,
    ))


class TestPresetValidation:
    @pytest.mark.asyncio
    async def test_compatible_packaging(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, packaging_type=PackagingType.BOTTLE)
        preset = _make_preset(
            compatible_packaging_types=[PackagingType.BOTTLE, PackagingType.BOX]
        )
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_incompatible_packaging(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, packaging_type=PackagingType.CAN)
        preset = _make_preset(
            compatible_packaging_types=[PackagingType.BOTTLE, PackagingType.BOX]
        )
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert len(errors) >= 1
        assert "not compatible" in errors[0]

    @pytest.mark.asyncio
    async def test_no_packaging_restriction(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, packaging_type=PackagingType.CAN)
        preset = _make_preset(compatible_packaging_types=None)
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_required_views_met(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
            angle_tag=AngleTag.FRONT,
        ))
        profile = _make_profile(
            product_id=product.id,
            views={AngleTag.FRONT: "a1"},
        )
        preset = _make_preset(required_views=[AngleTag.FRONT])
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_required_views_missing(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, views={AngleTag.BACK: "a1"})
        preset = _make_preset(required_views=[AngleTag.FRONT])
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert any("required views" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_preferred_views_missing_is_warning(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, views={AngleTag.FRONT: "a1"})
        preset = _make_preset(
            required_views=[],
            preferred_views=[AngleTag.FRONT, AngleTag.DETAIL],
        )
        warnings = await validate_compatibility(profile, preset, store, product.id)
        non_errors = [w for w in warnings if not w.startswith("ERROR:")]
        assert any("preferred" in w.lower() for w in non_errors)

    @pytest.mark.asyncio
    async def test_low_confidence_warning(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, ingest_confidence=0.2)
        preset = _make_preset()
        warnings = await validate_compatibility(profile, preset, store, product.id)
        assert any("confidence" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_good_confidence_no_warning(self, store: ProductStore) -> None:
        product = await store.create_product("Test")
        await _add_usable_asset(store, product.id)
        profile = _make_profile(product_id=product.id, ingest_confidence=0.8)
        preset = _make_preset()
        warnings = await validate_compatibility(profile, preset, store, product.id)
        assert not any("confidence" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_min_reference_count_enforced(self, store: ProductStore) -> None:
        """Real test: min_reference_count blocks when not enough usable assets."""
        product = await store.create_product("Test")
        profile = _make_profile(product_id=product.id)
        preset = _make_preset(
            min_reference_count=2,
            required_views=[],
        )

        # No assets at all — should fail
        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert any("usable reference" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_min_reference_count_passes_with_enough_assets(
        self, store: ProductStore
    ) -> None:
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
        ))
        await store.add_asset(ProductAsset(
            id="a2", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/2.jpg", width=800, height=600,
        ))

        profile = _make_profile(product_id=product.id)
        preset = _make_preset(min_reference_count=2, required_views=[])

        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert not any("usable reference" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_excluded_assets_not_counted(self, store: ProductStore) -> None:
        """Excluded assets don't count toward min_reference_count."""
        product = await store.create_product("Test")
        await store.add_asset(ProductAsset(
            id="a1", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/1.jpg", width=800, height=600,
        ))
        await store.add_asset(ProductAsset(
            id="a2", product_id=product.id, asset_type=AssetType.RAW_UPLOAD,
            file_path="/tmp/2.jpg", width=800, height=600,
            metadata={"excluded": True},
        ))

        profile = _make_profile(product_id=product.id)
        preset = _make_preset(min_reference_count=2, required_views=[])

        warnings = await validate_compatibility(profile, preset, store, product.id)
        errors = [w for w in warnings if w.startswith("ERROR:")]
        assert any("usable reference" in e.lower() for e in errors)
