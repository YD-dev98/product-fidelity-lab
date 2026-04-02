"""Tests for provider abstraction and strategy routing."""

from __future__ import annotations

from product_fidelity_lab.generation.strategy import (
    AdapterAssistedStrategy,
    ReferenceOnlyStrategy,
    route_strategy,
)
from product_fidelity_lab.models.preset import StudioPreset
from product_fidelity_lab.models.product import (
    ModelStatus,
    Product,
    ProductModel,
    ProviderMode,
)
from product_fidelity_lab.product.provider import get_provider


def _make_product(**kwargs: object) -> Product:
    defaults = {"id": "p1", "name": "Test"}
    defaults.update(kwargs)
    return Product(**defaults)  # type: ignore[arg-type]


def _make_preset(**kwargs: object) -> StudioPreset:
    defaults = {"preset_id": "test", "name": "Test"}
    defaults.update(kwargs)
    return StudioPreset(**defaults)  # type: ignore[arg-type]


def _make_model(**kwargs: object) -> ProductModel:
    defaults = {"product_id": "p1"}
    defaults.update(kwargs)
    return ProductModel(**defaults)  # type: ignore[arg-type]


class TestRouteStrategy:
    def test_no_model_uses_reference_only(self) -> None:
        product = _make_product()
        preset = _make_preset()
        strategy = route_strategy(product, preset, product_model=None)
        assert isinstance(strategy, ReferenceOnlyStrategy)

    def test_reference_only_model_uses_reference_only(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(mode=ProviderMode.REFERENCE_ONLY)
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, ReferenceOnlyStrategy)

    def test_adapter_ready_uses_adapter(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(
            mode=ProviderMode.ADAPTER,
            status=ModelStatus.READY,
            external_model_id="https://fal.ai/lora/abc123",
        )
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, AdapterAssistedStrategy)

    def test_adapter_training_uses_reference_only(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(
            mode=ProviderMode.ADAPTER,
            status=ModelStatus.TRAINING,
            external_model_id=None,
        )
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, ReferenceOnlyStrategy)

    def test_adapter_failed_uses_reference_only(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(
            mode=ProviderMode.ADAPTER,
            status=ModelStatus.FAILED,
        )
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, ReferenceOnlyStrategy)

    def test_adapter_no_model_id_uses_reference_only(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(
            mode=ProviderMode.ADAPTER,
            status=ModelStatus.READY,
            external_model_id=None,
        )
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, ReferenceOnlyStrategy)

    def test_finetune_ready_uses_adapter(self) -> None:
        product = _make_product()
        preset = _make_preset()
        model = _make_model(
            mode=ProviderMode.FINETUNE,
            status=ModelStatus.READY,
            external_model_id="https://replicate.com/model/xyz",
        )
        strategy = route_strategy(product, preset, product_model=model)
        assert isinstance(strategy, AdapterAssistedStrategy)


class TestGetProvider:
    def test_fal_provider(self) -> None:
        from unittest.mock import MagicMock

        fal = MagicMock()
        provider = get_provider("fal", fal)
        assert provider.provider_name == "fal"
        assert provider.mode == ProviderMode.ADAPTER

    def test_unknown_provider_raises(self) -> None:
        from unittest.mock import MagicMock

        import pytest

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider", MagicMock())


class TestProductModelLifecycle:
    def test_model_defaults(self) -> None:
        model = ProductModel(product_id="p1")
        assert model.status == ModelStatus.NONE
        assert model.mode == ProviderMode.REFERENCE_ONLY
        assert model.strength == 0.8
        assert model.trigger_word is None

    def test_ready_model(self) -> None:
        model = ProductModel(
            product_id="p1",
            provider="fal",
            mode=ProviderMode.ADAPTER,
            external_model_id="https://fal.ai/lora/abc",
            status=ModelStatus.READY,
            trained_on_n_images=5,
            trigger_word="PRODSHOT",
        )
        assert model.status == ModelStatus.READY
        assert model.trained_on_n_images == 5
