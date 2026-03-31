"""SQLite-backed product metadata store with asset folder management."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import aiosqlite
import structlog

from product_fidelity_lab.models.preset import StudioPreset
from product_fidelity_lab.models.product import (
    AngleTag,
    AssetType,
    Product,
    ProductAsset,
    ProductModel,
    ProductProfile,
    ProductStatus,
)

logger = structlog.get_logger()

SCHEMA_VERSION_KEY = "product_engine_schema_version"
CURRENT_SCHEMA_VERSION = "1"

CREATE_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    profile_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

CREATE_PRODUCT_ASSETS_TABLE = """
CREATE TABLE IF NOT EXISTS product_assets (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id),
    asset_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    fal_url TEXT,
    angle_tag TEXT NOT NULL DEFAULT 'unknown',
    width INTEGER,
    height INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
)
"""

CREATE_PRODUCT_ASSETS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_product_assets_product ON product_assets(product_id)
"""

CREATE_PRODUCT_MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS product_models (
    product_id TEXT PRIMARY KEY REFERENCES products(id),
    provider TEXT NOT NULL DEFAULT 'fal',
    mode TEXT NOT NULL DEFAULT 'reference_only',
    external_model_id TEXT,
    strength REAL NOT NULL DEFAULT 0.8,
    status TEXT NOT NULL DEFAULT 'none',
    trained_on_n_images INTEGER NOT NULL DEFAULT 0,
    trigger_word TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

CREATE_STUDIO_PRESETS_TABLE = """
CREATE TABLE IF NOT EXISTS studio_presets (
    preset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL,
    is_builtin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
)
"""


def is_usable_reference(
    asset: ProductAsset,
    required_views: list[AngleTag] | None = None,
) -> bool:
    """Check if an asset qualifies as a usable visual reference.

    A usable visual reference satisfies ALL of:
    1. asset_type in {raw_upload, cleaned}
    2. Image dimensions are valid (width > 0, height > 0)
    3. Not manually excluded (no 'excluded: true' in metadata)
    4. If required_views is specified: asset has a matching angle_tag (not unknown)
    """
    if asset.asset_type not in (AssetType.RAW_UPLOAD, AssetType.CLEANED):
        return False
    if not asset.width or not asset.height or asset.width <= 0 or asset.height <= 0:
        return False
    if asset.metadata.get("excluded"):
        return False
    if required_views and asset.angle_tag == AngleTag.UNKNOWN:
        return False
    return not (required_views and asset.angle_tag not in required_views)


class ProductStore:
    """Manages product metadata in SQLite and asset files on disk."""

    def __init__(self, db_path: Path, products_dir: Path) -> None:
        self._db_path = db_path
        self._products_dir = products_dir
        self._products_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Create tables if needed and check schema version."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(CREATE_SCHEMA_META)
            await db.execute(CREATE_PRODUCTS_TABLE)
            await db.execute(CREATE_PRODUCT_ASSETS_TABLE)
            await db.execute(CREATE_PRODUCT_ASSETS_INDEX)
            await db.execute(CREATE_PRODUCT_MODELS_TABLE)
            await db.execute(CREATE_STUDIO_PRESETS_TABLE)
            await db.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                (SCHEMA_VERSION_KEY, CURRENT_SCHEMA_VERSION),
            )
            await db.commit()
        logger.info("product_store.initialized", db_path=str(self._db_path))

    # ── Products ──────────────────────────────────────────────────────

    async def create_product(self, name: str) -> Product:
        product_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        product = Product(
            id=product_id,
            name=name,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )
        # Create directories
        (self._products_dir / product_id / "uploads").mkdir(parents=True, exist_ok=True)
        (self._products_dir / product_id / "assets").mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO products (id, name, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (product_id, name, product.status, now, now),
            )
            await db.commit()
        logger.info("product_store.created_product", product_id=product_id, name=name)
        return product

    async def get_product(self, product_id: str) -> Product | None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM products WHERE id = ?", (product_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_product(row)

    async def list_products(self, limit: int = 50) -> list[Product]:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM products ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [_row_to_product(row) for row in rows]

    async def update_product(
        self,
        product_id: str,
        *,
        status: ProductStatus | None = None,
        profile: ProductProfile | None = None,
    ) -> None:
        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now_iso()]

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)
        if profile is not None:
            updates.append("profile_json = ?")
            params.append(profile.model_dump_json())

        params.append(product_id)
        sql = f"UPDATE products SET {', '.join(updates)} WHERE id = ?"
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(sql, params)
            await db.commit()

    async def delete_product(self, product_id: str) -> None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "DELETE FROM product_models WHERE product_id = ?", (product_id,)
            )
            await db.execute(
                "DELETE FROM product_assets WHERE product_id = ?", (product_id,)
            )
            await db.execute("DELETE FROM products WHERE id = ?", (product_id,))
            await db.commit()
        product_dir = self._products_dir / product_id
        if product_dir.exists():
            shutil.rmtree(product_dir)

    async def get_profile(self, product_id: str) -> ProductProfile | None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT profile_json FROM products WHERE id = ?", (product_id,)
            )
            row = await cursor.fetchone()
            if row is None or row[0] is None:
                return None
            return ProductProfile.model_validate_json(row[0])

    # ── Assets ────────────────────────────────────────────────────────

    async def add_asset(self, asset: ProductAsset) -> ProductAsset:
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO product_assets "
                "(id, product_id, asset_type, file_path, fal_url, angle_tag, "
                "width, height, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset.id,
                    asset.product_id,
                    asset.asset_type,
                    asset.file_path,
                    asset.fal_url,
                    asset.angle_tag,
                    asset.width,
                    asset.height,
                    json.dumps(asset.metadata),
                    asset.created_at.isoformat(),
                ),
            )
            await db.commit()
        return asset

    async def get_assets(
        self,
        product_id: str,
        asset_type: AssetType | None = None,
    ) -> list[ProductAsset]:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            if asset_type:
                cursor = await db.execute(
                    "SELECT * FROM product_assets WHERE product_id = ? AND asset_type = ? "
                    "ORDER BY created_at",
                    (product_id, asset_type),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM product_assets WHERE product_id = ? ORDER BY created_at",
                    (product_id,),
                )
            rows = await cursor.fetchall()
            return [_row_to_asset(row) for row in rows]

    async def get_asset(self, asset_id: str) -> ProductAsset | None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM product_assets WHERE id = ?", (asset_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_asset(row)

    async def update_asset(
        self,
        asset_id: str,
        *,
        fal_url: str | None = None,
        angle_tag: AngleTag | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if fal_url is not None:
            updates.append("fal_url = ?")
            params.append(fal_url)
        if angle_tag is not None:
            updates.append("angle_tag = ?")
            params.append(angle_tag.value)
        if not updates:
            return
        params.append(asset_id)
        sql = f"UPDATE product_assets SET {', '.join(updates)} WHERE id = ?"
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(sql, params)
            await db.commit()

    # ── Models ────────────────────────────────────────────────────────

    async def upsert_model(self, model: ProductModel) -> None:
        now = _now_iso()
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO product_models "
                "(product_id, provider, mode, external_model_id, strength, "
                "status, trained_on_n_images, trigger_word, metadata_json, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(product_id) DO UPDATE SET "
                "provider=excluded.provider, mode=excluded.mode, "
                "external_model_id=excluded.external_model_id, "
                "strength=excluded.strength, status=excluded.status, "
                "trained_on_n_images=excluded.trained_on_n_images, "
                "trigger_word=excluded.trigger_word, "
                "metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
                (
                    model.product_id,
                    model.provider,
                    model.mode,
                    model.external_model_id,
                    model.strength,
                    model.status,
                    model.trained_on_n_images,
                    model.trigger_word,
                    json.dumps(model.metadata),
                    now,
                    now,
                ),
            )
            await db.commit()

    async def get_model(self, product_id: str) -> ProductModel | None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM product_models WHERE product_id = ?", (product_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _row_to_model(row)

    # ── Presets ───────────────────────────────────────────────────────

    async def save_preset(
        self, preset: StudioPreset, *, is_builtin: bool = False
    ) -> None:
        now = _now_iso()
        config = preset.model_dump(mode="json")
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(
                "INSERT INTO studio_presets "
                "(preset_id, name, description, config_json, is_builtin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(preset_id) DO UPDATE SET "
                "name=excluded.name, description=excluded.description, "
                "config_json=excluded.config_json, is_builtin=excluded.is_builtin",
                (
                    preset.preset_id,
                    preset.name,
                    preset.description,
                    json.dumps(config),
                    int(is_builtin),
                    now,
                ),
            )
            await db.commit()

    async def get_preset(self, preset_id: str) -> StudioPreset | None:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM studio_presets WHERE preset_id = ?", (preset_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return StudioPreset.model_validate(json.loads(row["config_json"]))

    async def list_presets(self) -> list[StudioPreset]:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM studio_presets ORDER BY is_builtin DESC, name"
            )
            rows = await cursor.fetchall()
            return [
                StudioPreset.model_validate(json.loads(row["config_json"]))
                for row in rows
            ]

    async def seed_builtin_presets(self, presets: dict[str, StudioPreset]) -> None:
        """Upsert all builtin presets."""
        for preset in presets.values():
            await self.save_preset(preset, is_builtin=True)
        logger.info("product_store.seeded_presets", count=len(presets))

    # ── Directory helpers ─────────────────────────────────────────────

    def product_dir(self, product_id: str) -> Path:
        d = self._products_dir / product_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def uploads_dir(self, product_id: str) -> Path:
        d = self._products_dir / product_id / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def assets_dir(self, product_id: str) -> Path:
        d = self._products_dir / product_id / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _now_iso() -> str:
    return datetime.now().isoformat()


def _row_to_product(row: Any) -> Product:
    return Product(
        id=row["id"],
        name=row["name"],
        status=ProductStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_asset(row: Any) -> ProductAsset:
    return ProductAsset(
        id=row["id"],
        product_id=row["product_id"],
        asset_type=AssetType(row["asset_type"]),
        file_path=row["file_path"],
        fal_url=row["fal_url"],
        angle_tag=AngleTag(row["angle_tag"]),
        width=row["width"],
        height=row["height"],
        metadata=json.loads(row["metadata_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_model(row: Any) -> ProductModel:
    from product_fidelity_lab.models.product import ModelStatus, ProviderMode

    return ProductModel(
        product_id=row["product_id"],
        provider=row["provider"],
        mode=ProviderMode(row["mode"]),
        external_model_id=row["external_model_id"],
        strength=row["strength"],
        status=ModelStatus(row["status"]),
        trained_on_n_images=row["trained_on_n_images"],
        trigger_word=row["trigger_word"],
        metadata=json.loads(row["metadata_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
