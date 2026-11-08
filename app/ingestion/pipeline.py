"""
Full ingestion pipeline (Legal In-Memory Architecture).

Source strategy:
  1. Official 30s previews via Apple / iTunes Search API & Deezer.
  2. Direct in-memory decoding with PyAV (zero MP3 disk writes).
  3. Landmark constellation fingerprinting for instant Shazam-style identification.
  4. 512-dim semantic multimodal embedding upserted to Qdrant.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

from app.config import settings
from app.database.fingerprint import get_fingerprinter
from app.database.qdrant import song_exists, upsert_song
from app.ingestion.extractor import extract_array_embedding
from app.ingestion.preview_client import (
    PreviewTrack,
    fetch_preview_candidates,
)

logger = logging.getLogger(__name__)


async def ingest_preview_track(track: PreviewTrack) -> bool:
    """
    Ingest a single legal PreviewTrack in-memory:
    1. Landmark constellation fingerprinting -> SQLite.
    2. Semantic vector embedding -> Qdrant.
    """
    if song_exists(track.track_id):
        logger.info("Skip (already stored in Qdrant): %s – %s", track.artist, track.title)
        return False

    try:
        # Step 1: Acoustic landmark fingerprinting (for instant snippet matching)
        fingerprinter = get_fingerprinter()
        fingerprinter.index_song(
            song_id=track.track_id,
            y=track.audio_samples,
            meta={
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "store_url": track.store_url,
                "cover_url": track.cover_url,
            },
        )

        # Step 2: Semantic embedding vector (CLAP 512-dim)
        vector = await extract_array_embedding(track.audio_samples)

        # Step 3: Upsert into Qdrant
        upsert_song(
            vector=vector,
            title=track.title,
            artist=track.artist,
            album=track.album,
            track_id=track.track_id,
            store_url=track.store_url,
            cover_url=track.cover_url,
            source=track.source,
            duration_s=track.duration_s,
        )

        logger.info(
            "Successfully ingested: %s – %s [%s]",
            track.artist,
            track.title,
            track.source,
        )
        return True

    except Exception as exc:
        logger.error(
            "Failed to ingest track %s (%s): %s",
            track.track_id,
            track.title,
            exc,
            exc_info=True,
        )
        return False


async def run_ingestion_cycle(n_songs: int = settings.SONGS_PER_RUN) -> dict[str, Any]:
    """
    One full legal ingestion cycle:
    1. Fetch N candidates from preview APIs.
    2. Filter already stored tracks.
    3. Ingest each into Fingerprinter + Qdrant.
    4. Return summary.
    """
    candidates = await fetch_preview_candidates(n=n_songs * 2)
    if not candidates:
        logger.warning("No candidate preview tracks returned.")
        return {"source": "previews", "ingested": 0, "skipped": 0, "errors": 0}

    ingested = 0
    skipped = 0
    errors = 0

    for track in candidates:
        if ingested >= n_songs:
            break

        if song_exists(track.track_id):
            skipped += 1
            continue

        success = await ingest_preview_track(track)
        if success:
            ingested += 1
        else:
            errors += 1

        # Garbage collect any unreferenced audio buffers
        gc.collect()

    summary = {
        "source": "legal_previews",
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("Ingestion cycle complete: %s", summary)
    return summary


# Compatibility wrapper for single-track manual trigger
async def ingest_song(track_id: str, meta: Any = None) -> bool:
    """Manually ingest a track by search query or ID."""
    query = f"{meta.artist} {meta.title}" if meta else track_id
    tracks = await fetch_preview_candidates(n=1, query=query)
    if not tracks:
        logger.warning("Could not find preview for: %s", query)
        return False
    return await ingest_preview_track(tracks[0])
