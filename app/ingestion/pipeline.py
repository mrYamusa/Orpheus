"""
Full ingestion pipeline for a single song.

Steps:
  1. Skip if the YouTube ID is already in Qdrant.
  2. Download the audio to a temporary scratch directory.
  3. Extract all features (VibeNet + librosa).
  4. Build the embedding vector.
  5. Upsert into Qdrant.
  6. Delete the local audio file immediately (scratch-only).

The public entry point is `ingest_song(video_id, meta)` which is called
by the scheduler.  There is also `run_ingestion_cycle()` which picks
random queries, searches YouTube, and ingests N new songs per run.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from app.config import settings
from app.database.qdrant import song_exists, upsert_frames, upsert_song
from app.ingestion.downloader import VideoMeta, download_song, search_youtube
from app.ingestion.extractor import extract_all

logger = logging.getLogger(__name__)


async def ingest_song(video_id: str, meta: VideoMeta) -> bool:
    """
    Download, extract, embed, and store a single song.

    Returns True on success, False if skipped or errored.
    """
    # --- Guard: already in DB ---
    if song_exists(video_id):
        logger.info("Skip (already stored): %s – %s", meta.artist, meta.title)
        return False

    audio_path: Path | None = None
    try:
        # --- Download ---
        logger.info("Downloading: %s – %s (%s)", meta.artist, meta.title, video_id)
        audio_path, refreshed_meta = await download_song(video_id, settings.SCRATCH_DIR)

        # Use refreshed meta from yt-dlp (may have better title/artist tags)
        final_meta = refreshed_meta if refreshed_meta.title != "Unknown Title" else meta

        # --- Extract song-level features + timestamped frames in one pass ---
        song_features, frame_list = await extract_all(audio_path)

        # --- Store song point in Qdrant ---
        song_point_id = upsert_song(
            vector=song_features.to_embedding(),
            title=final_meta.title,
            artist=final_meta.artist,
            youtube_id=video_id,
            duration_s=song_features.duration_s,
            tempo_bpm=song_features.tempo_bpm,
            key=song_features.key,
            mode=song_features.mode,
            acousticness=song_features.acousticness,
            danceability=song_features.danceability,
            energy=song_features.energy,
            instrumentalness=song_features.instrumentalness,
            liveness=song_features.liveness,
            speechiness=song_features.speechiness,
            valence=song_features.valence,
            rms_mean=song_features.rms_mean,
            spectral_centroid_mean=song_features.spectral_centroid_mean,
            spectral_bandwidth_mean=song_features.spectral_bandwidth_mean,
            spectral_rolloff_mean=song_features.spectral_rolloff_mean,
            zcr_mean=song_features.zcr_mean,
            mfcc_means=song_features.mfcc_means,
        )

        # --- Store frame points (timestamps + local embeddings) ---
        n_frames = upsert_frames(
            song_id=song_point_id,
            youtube_id=video_id,
            title=final_meta.title,
            artist=final_meta.artist,
            frames=frame_list,
        )

        logger.info(
            "Ingested: %s – %s (%d frames @ %.1fs windows)",
            final_meta.artist,
            final_meta.title,
            n_frames,
            settings.FRAME_WINDOW_S,
        )
        return True

    except Exception as exc:
        logger.error("Failed to ingest %s: %s", video_id, exc, exc_info=True)
        return False

    finally:
        # Always delete the local file, even on failure
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)
            logger.debug("Deleted scratch file: %s", audio_path)


async def run_ingestion_cycle(n_songs: int = settings.SONGS_PER_RUN) -> dict:
    """
    One full ingestion cycle:
      - Pick a random query from the pool.
      - Search YouTube for candidates.
      - Ingest up to n_songs new (not-yet-stored) songs.
      - Return a summary dict.
    """
    query = random.choice(settings.YOUTUBE_QUERY_POOL)
    logger.info(
        "Ingestion cycle started — query: '%s', target: %d songs", query, n_songs
    )

    # Search for more candidates than we need in case some are already stored
    candidates = await search_youtube(query, max_results=n_songs * 3)

    if not candidates:
        logger.warning("YouTube search returned no results for query: '%s'", query)
        return {"query": query, "ingested": 0, "skipped": 0, "errors": 0}

    ingested = 0
    skipped = 0
    errors = 0

    for candidate in candidates:
        if ingested >= n_songs:
            break
        result = await ingest_song(candidate.id, candidate)
        if result is True:
            ingested += 1
        elif result is False:
            skipped += 1
        else:
            errors += 1

    summary = {
        "query": query,
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("Ingestion cycle complete: %s", summary)
    return summary
