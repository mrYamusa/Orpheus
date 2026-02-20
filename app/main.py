"""
Orpheus FastAPI application factory.

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up
"""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from app.api.routes import router
from app.config import settings
from app.database.qdrant import ensure_collection, ensure_frame_collection
from app.scheduler.jobs import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info("Orpheus starting up…")

    # Ensure scratch directory exists
    settings.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Scratch dir: %s", settings.SCRATCH_DIR.resolve())

    # Bootstrap Qdrant collections
    try:
        ensure_collection()
        ensure_frame_collection()
    except Exception as exc:
        logger.warning(
            "Qdrant not reachable on startup: %s — will retry on first request.", exc
        )

    # Start background ingestion scheduler
    start_scheduler()

    yield  # ── App is running ──

    # ── Shutdown ─────────────────────────────────────────────
    stop_scheduler()
    logger.info("Orpheus shut down cleanly.")


app = FastAPI(
    title="Orpheus",
    description=(
        "Voice-activated music discovery engine.\n\n"
        "Ingests songs from YouTube, extracts audio features, embeds them "
        "in a Qdrant vector database, and serves semantic playlist search."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/scalar", include_in_schema=False)
async def scalar_docs():
    """Interactive API reference powered by Scalar."""
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
