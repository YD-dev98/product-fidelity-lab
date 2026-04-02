"""Tests for repair/compositing logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from product_fidelity_lab.models.preset import (
    CameraAngle,
    Candidate,
    RepairAction,
    StudioPreset,
)
from product_fidelity_lab.models.product import (
    AssetType,
    ProductAsset,
    ProductProfile,
)
from product_fidelity_lab.repair.composite import (
    is_repair_eligible,
    plan_repair,
)
from product_fidelity_lab.storage.product_store import ProductStore


def _make_preset(**kwargs: object) -> StudioPreset:
    defaults = {"preset_id": "test", "name": "Test"}
    defaults.update(kwargs)
    return StudioPreset(**defaults)  # type: ignore[arg-type]


def _make_profile(**kwargs: object) -> ProductProfile:
    defaults = {"product_id": "p1"}
    defaults.update(kwargs)
    return ProductProfile(**defaults)  # type: ignore[arg-type]


def _make_candidate(**kwargs: object) -> Candidate:
    defaults = {
        "candidate_id": "c1",
        "image_url": "https://example.com/img.jpg",
        "seed": 42,
        "model_id": "flux",
        "strategy_used": "reference_only",
        "generation_ms": 1000,
        "cost_estimate": 0.05,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)  # type: ignore[arg-type]


class TestRepairEligibility:
    def test_front_facing_text_critical_eligible(self) -> None:
        preset = _make_preset(
            camera_angle=CameraAngle.FRONT,
            supports_text_critical=True,
        )
        assert is_repair_eligible(preset) is True

    def test_eye_level_text_critical_eligible(self) -> None:
        preset = _make_preset(
            camera_angle=CameraAngle.EYE_LEVEL,
            supports_text_critical=True,
        )
        assert is_repair_eligible(preset) is True

    def test_angled_not_eligible(self) -> None:
        preset = _make_preset(
            camera_angle=CameraAngle.FRONT_LEFT_45,
            supports_text_critical=True,
        )
        assert is_repair_eligible(preset) is False

    def test_not_text_critical_not_eligible(self) -> None:
        preset = _make_preset(
            camera_angle=CameraAngle.FRONT,
            supports_text_critical=False,
        )
        assert is_repair_eligible(preset) is False

    def test_overhead_not_eligible(self) -> None:
        preset = _make_preset(
            camera_angle=CameraAngle.OVERHEAD_45,
            supports_text_critical=True,
        )
        assert is_repair_eligible(preset) is False


class TestPlanRepair:
    @pytest.mark.asyncio
    async def test_no_repair_needed(self) -> None:
        candidate = _make_candidate(
            label_score=0.9,
            identity_score=0.9,
            style_score=0.9,
            composition_score=0.9,
        )
        profile = _make_profile()
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert plan.actions == []

    @pytest.mark.asyncio
    async def test_logo_repair_on_low_label(self) -> None:
        candidate = _make_candidate(label_score=0.3)
        profile = _make_profile(logo_asset_id="logo1")
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOGO_COMPOSITE in plan.actions
        assert plan.logo_asset_id == "logo1"

    @pytest.mark.asyncio
    async def test_label_repair_on_very_low_label(self) -> None:
        candidate = _make_candidate(label_score=0.2)
        profile = _make_profile(label_asset_id="label1")
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LABEL_COMPOSITE in plan.actions
        assert plan.label_asset_id == "label1"

    @pytest.mark.asyncio
    async def test_critical_text_absent_triggers_repair(self) -> None:
        candidate = _make_candidate(
            filter_reasons=["critical_text_absent"],
        )
        profile = _make_profile(
            logo_asset_id="logo1",
            label_asset_id="label1",
        )
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOGO_COMPOSITE in plan.actions
        assert RepairAction.LABEL_COMPOSITE in plan.actions

    @pytest.mark.asyncio
    async def test_localized_edit_on_low_style(self) -> None:
        candidate = _make_candidate(
            identity_score=0.8,
            style_score=0.3,
            composition_score=0.7,
        )
        profile = _make_profile()
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOCALIZED_EDIT in plan.actions
        assert plan.edit_prompt is not None

    @pytest.mark.asyncio
    async def test_no_localized_edit_if_identity_low(self) -> None:
        """Don't edit if identity is bad — the whole image is wrong."""
        candidate = _make_candidate(
            identity_score=0.3,
            style_score=0.3,
        )
        profile = _make_profile()
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOCALIZED_EDIT not in plan.actions

    @pytest.mark.asyncio
    async def test_no_logo_repair_without_asset(self) -> None:
        """No logo asset = no logo composite, even if label score is low."""
        candidate = _make_candidate(label_score=0.3)
        profile = _make_profile(logo_asset_id=None)
        preset = _make_preset()

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOGO_COMPOSITE not in plan.actions


class TestIngestToRepairIntegration:
    """Verify profiles with ingested logo/label assets reach composite repair."""

    @pytest.fixture
    async def store(self, tmp_path: Path) -> ProductStore:
        s = ProductStore(
            db_path=tmp_path / "test.db",
            products_dir=tmp_path / "products",
        )
        await s.initialize()
        return s

    @pytest.mark.asyncio
    async def test_profile_with_logo_asset_enables_repair(
        self, store: ProductStore,
    ) -> None:
        """A profile with logo_asset_id from ingest enables logo composite."""
        product = await store.create_product("Test")

        # Simulate what ingest does: create a logo_crop asset with bbox
        logo_asset = ProductAsset(
            id="logo1",
            product_id=product.id,
            asset_type=AssetType.LOGO_CROP,
            file_path="/tmp/logo.png",
            fal_url="https://fal.ai/logo.png",
            width=200,
            height=80,
            metadata={
                "source_asset_id": "raw1",
                "bbox": {"x": 0.3, "y": 0.1, "width": 0.4, "height": 0.15},
            },
        )
        await store.add_asset(logo_asset)

        # Build profile the way ingest would
        profile = _make_profile(
            product_id=product.id,
            logo_asset_id="logo1",
        )
        await store.update_product(product.id, profile=profile)

        # Candidate with low label score
        candidate = _make_candidate(label_score=0.3)
        preset = _make_preset(
            camera_angle=CameraAngle.FRONT,
            supports_text_critical=True,
        )

        assert is_repair_eligible(preset)

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOGO_COMPOSITE in plan.actions
        assert plan.logo_asset_id == "logo1"

        # Verify the asset has bbox metadata for composite placement
        fetched = await store.get_asset("logo1")
        assert fetched is not None
        assert "bbox" in fetched.metadata
        assert fetched.metadata["bbox"]["x"] == 0.3

    @pytest.mark.asyncio
    async def test_profile_without_crops_skips_composite(
        self, store: ProductStore,
    ) -> None:
        """A profile with no logo/label assets falls back to edit-only."""
        product = await store.create_product("Test")
        profile = _make_profile(
            product_id=product.id,
            logo_asset_id=None,
            label_asset_id=None,
        )
        await store.update_product(product.id, profile=profile)

        candidate = _make_candidate(label_score=0.3, style_score=0.3, identity_score=0.8)
        preset = _make_preset(
            camera_angle=CameraAngle.FRONT,
            supports_text_critical=True,
        )

        plan = await plan_repair(candidate, profile, preset)
        assert RepairAction.LOGO_COMPOSITE not in plan.actions
        assert RepairAction.LABEL_COMPOSITE not in plan.actions
        # May still get localized edit if style is low
        assert RepairAction.LOCALIZED_EDIT in plan.actions
