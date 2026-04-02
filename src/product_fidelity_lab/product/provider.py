"""Provider abstraction for product model training and inference.

Supports multiple backends (fal LoRA, Replicate, BFL) behind a
common interface. Phase 4 ships with FalLoraProvider only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import structlog

from product_fidelity_lab.models.product import (
    ModelStatus,
    ProductModel,
    ProviderMode,
)

if TYPE_CHECKING:
    from product_fidelity_lab.generation.client import FalClient
    from product_fidelity_lab.storage.product_store import ProductStore

logger = structlog.get_logger()

FAL_LORA_TRAINING_MODEL = "fal-ai/flux-lora-fast-training"
FAL_LORA_INFERENCE_MODEL = "fal-ai/flux-lora"


class ProviderBackend(Protocol):
    """Protocol for provider backends."""

    async def start_training(
        self,
        product_id: str,
        image_urls: list[str],
        trigger_word: str,
        product_store: ProductStore,
    ) -> ProductModel: ...

    async def check_status(
        self,
        model: ProductModel,
    ) -> ProductModel: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def mode(self) -> ProviderMode: ...


class FalLoraProvider:
    """LoRA fine-tuning via fal.ai's flux-lora-fast-training."""

    def __init__(self, fal_client: FalClient) -> None:
        self._fal = fal_client

    @property
    def provider_name(self) -> str:
        return "fal"

    @property
    def mode(self) -> ProviderMode:
        return ProviderMode.ADAPTER

    async def start_training(
        self,
        product_id: str,
        image_urls: list[str],
        trigger_word: str,
        product_store: ProductStore,
    ) -> ProductModel:
        """Start LoRA training on fal.ai.

        Submits training job and returns a ProductModel with status=TRAINING.
        The external_model_id is populated when training completes.
        """
        # fal training requires images_data_url pointing to a zip
        zip_url = await _create_training_zip(image_urls)

        input_data: dict[str, Any] = {
            "images_data_url": zip_url,
            "trigger_word": trigger_word,
            "steps": 1000,
            "rank": 16,
        }

        # Persist TRAINING status before the blocking call
        training_model = ProductModel(
            product_id=product_id,
            provider=self.provider_name,
            mode=self.mode,
            status=ModelStatus.TRAINING,
            trained_on_n_images=len(image_urls),
            trigger_word=trigger_word,
            metadata={"training_model": FAL_LORA_TRAINING_MODEL},
        )
        await product_store.upsert_model(training_model)
        logger.info("provider.training_started", product_id=product_id)

        try:
            result = await self._fal.subscribe(
                FAL_LORA_TRAINING_MODEL, input_data,
            )

            # fal returns the LoRA weights URL as the model ID
            lora_url = str(result.get("diffusers_lora_file", {}).get("url", ""))
            if not lora_url:
                lora_url = str(result.get("config_file", {}).get("url", ""))

            model = ProductModel(
                product_id=product_id,
                provider=self.provider_name,
                mode=self.mode,
                external_model_id=lora_url or None,
                status=ModelStatus.READY if lora_url else ModelStatus.FAILED,
                trained_on_n_images=len(image_urls),
                trigger_word=trigger_word,
                metadata={
                    "training_model": FAL_LORA_TRAINING_MODEL,
                    "steps": 1000,
                    "rank": 16,
                },
            )

            await product_store.upsert_model(model)

            logger.info(
                "provider.training_complete",
                product_id=product_id,
                status=model.status,
                lora_url=lora_url[:80] if lora_url else None,
            )
            return model

        except Exception as exc:
            logger.error(
                "provider.training_failed",
                product_id=product_id,
                error=str(exc),
            )
            model = ProductModel(
                product_id=product_id,
                provider=self.provider_name,
                mode=self.mode,
                status=ModelStatus.FAILED,
                trained_on_n_images=len(image_urls),
                trigger_word=trigger_word,
                metadata={"error": str(exc)},
            )
            await product_store.upsert_model(model)
            return model

    async def check_status(self, model: ProductModel) -> ProductModel:
        """Check training status. For fal, training is synchronous via subscribe."""
        # fal's subscribe blocks until done, so status is already final
        return model


def get_provider(
    provider_name: str,
    fal_client: FalClient,
) -> ProviderBackend:
    """Get a provider backend by name."""
    providers: dict[str, type] = {
        "fal": FalLoraProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        msg = f"Unknown provider: {provider_name}"
        raise ValueError(msg)
    return cls(fal_client)  # type: ignore[call-arg]


async def _create_training_zip(image_urls: list[str]) -> str:
    """Download images, zip them, and upload the zip to fal storage.

    fal's training API requires a single images_data_url pointing to a zip.
    """
    import tempfile
    import zipfile
    from pathlib import Path

    import fal_client
    import httpx

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "training_images.zip"

        async with httpx.AsyncClient(timeout=60) as client:
            with zipfile.ZipFile(zip_path, "w") as zf:
                for i, url in enumerate(image_urls):
                    resp = await client.get(url)
                    resp.raise_for_status()
                    # Determine extension from content-type
                    ct = resp.headers.get("content-type", "image/jpeg")
                    ext = {
                        "image/jpeg": ".jpg",
                        "image/png": ".png",
                        "image/webp": ".webp",
                    }.get(ct, ".jpg")
                    zf.writestr(f"image_{i:03d}{ext}", resp.content)

        zip_url: str = await fal_client.upload_file_async(zip_path)  # type: ignore[reportUnknownMemberType]
        logger.info("provider.training_zip_uploaded", url=zip_url[:80], count=len(image_urls))
        return zip_url
