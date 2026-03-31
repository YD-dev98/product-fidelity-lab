"""Product Fidelity Lab — Product Photography AI Evaluation System."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from product_fidelity_lab.config import get_settings
from product_fidelity_lab.storage.product_store import ProductStore
from product_fidelity_lab.storage.replay import ReplayStore
from product_fidelity_lab.storage.run_store import RunStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

_run_store: RunStore | None = None
_replay_store: ReplayStore | None = None
_product_store: ProductStore | None = None


def get_run_store() -> RunStore:
    assert _run_store is not None, "RunStore not initialized"
    return _run_store


def get_replay_store() -> ReplayStore | None:
    return _replay_store


def get_product_store() -> ProductStore:
    assert _product_store is not None, "ProductStore not initialized"
    return _product_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global _run_store, _replay_store, _product_store  # noqa: PLW0603
    settings = get_settings()

    # Propagate keys to os.environ for third-party libs (fal-client) — only in live mode
    if settings.live_ready:
        os.environ.setdefault("FAL_KEY", settings.fal_key)

    # Initialize RunStore
    _run_store = RunStore(
        db_path=settings.db_path,
        runs_dir=settings.data_dir / "runs",
    )
    await _run_store.initialize()
    recovered = await _run_store.recover_interrupted()
    if recovered:
        logger.info("startup.recovered_interrupted_runs", count=recovered)

    # Initialize ProductStore
    _product_store = ProductStore(
        db_path=settings.db_path,
        products_dir=settings.products_dir,
    )
    await _product_store.initialize()
    from product_fidelity_lab.generation.presets import BUILTIN_PRESETS

    await _product_store.seed_builtin_presets(BUILTIN_PRESETS)

    # Initialize ReplayStore
    replay_dir = settings.data_dir / "replay"
    _replay_store = ReplayStore(replay_dir)

    mode = "replay" if settings.replay_mode else "live"
    logger.info(
        "startup.complete",
        mode=mode,
        data_dir=str(settings.data_dir),
        live_ready=settings.live_ready,
        replay_available=_replay_store.available,
    )
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Product Fidelity Lab",
    description="Product Photography AI Evaluation System",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount API routers
from product_fidelity_lab.api.evaluate import router as evaluate_router  # noqa: E402
from product_fidelity_lab.api.generate import router as generate_router  # noqa: E402
from product_fidelity_lab.api.golden import router as golden_router  # noqa: E402
from product_fidelity_lab.api.products import router as products_router  # noqa: E402
from product_fidelity_lab.api.render import router as render_router  # noqa: E402

app.include_router(evaluate_router)
app.include_router(generate_router)
app.include_router(golden_router)
app.include_router(products_router)
app.include_router(render_router)

# Serve frontend
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/mode")
async def mode() -> dict[str, Any]:
    """Report server mode so the frontend can adapt."""
    settings = get_settings()
    store = get_replay_store()
    return {
        "mode": "replay" if settings.replay_mode else "live",
        "live_ready": settings.live_ready,
        "replay_available": store.available if store else False,
    }


@app.get("/api/replay/runs")
async def replay_list() -> list[dict[str, Any]]:
    """List precomputed replay runs."""
    store = get_replay_store()
    if store is None or not store.available:
        return []
    return store.list_runs()


@app.get("/api/replay/runs/{run_id}")
async def replay_get(run_id: str) -> Any:
    """Get a specific replay run."""
    store = get_replay_store()
    if store is None:
        return JSONResponse(status_code=404, content={"detail": "No replay data"})
    run = store.get_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "Replay run not found"})
    return run


@app.get("/")
async def index() -> Any:
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Product Fidelity Lab API — frontend not built yet"})


def cli() -> None:
    """CLI entrypoint for pfl-demo."""
    replay = "--replay" in sys.argv

    if replay:
        os.environ.setdefault("PFL_REPLAY_MODE", "1")
        logger.info("Starting Product Fidelity Lab in replay/demo mode")
    else:
        logger.info("Starting Product Fidelity Lab in live mode")

    settings = get_settings()

    if not replay and not settings.live_ready:
        logger.warning("Live mode but API keys not configured — use --replay for demo mode")

    uvicorn.run(
        "product_fidelity_lab.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
