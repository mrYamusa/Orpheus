"""
Background scheduler.

Uses APScheduler's AsyncIOScheduler so it runs inside the same
asyncio event loop as FastAPI — no extra threads needed.

The job fires every SCHEDULE_HOURS hours and calls
run_ingestion_cycle() to download and embed N new songs.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.ingestion.pipeline import run_ingestion_cycle

logger = logging.getLogger(__name__)

# Module-level singleton
_scheduler: AsyncIOScheduler | None = None


async def _ingestion_job() -> None:
    """Wrapper so APScheduler can call the async pipeline."""
    logger.info("Scheduled ingestion job triggered.")
    try:
        summary = await run_ingestion_cycle()
        logger.info("Job summary: %s", summary)
    except Exception as exc:
        logger.error("Ingestion job crashed: %s", exc, exc_info=True)


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="UTC")
        _scheduler.add_job(
            _ingestion_job,
            trigger=IntervalTrigger(minutes=settings.SCHEDULE_MINUTES),
            id="ingestion_job",
            name=f"Ingest {settings.SONGS_PER_RUN} songs every {settings.SCHEDULE_MINUTES}min",
            replace_existing=True,
            misfire_grace_time=60 * 5,  # 5-minute grace window
        )
        logger.info(
            "Scheduler configured: every %d min, %d song(s) per run.",
            settings.SCHEDULE_MINUTES,
            settings.SONGS_PER_RUN,
        )
    return _scheduler


def start_scheduler() -> None:
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("Scheduler started.")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
