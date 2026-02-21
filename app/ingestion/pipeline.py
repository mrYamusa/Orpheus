"""
Full ingestion pipeline.

Source selection (in order of preference):
  1. Spotify trending tracks (if SPOTIFY_CLIENT_ID / SECRET are set)
     → category randomly picked from SPOTIFY_CATEGORY_POOL
     → each track looked up on YouTube by "{artist} {title} official audio"
  2. YouTube direct search (fallback when Spotify creds are missing)

Batch behaviour:
  All N songs are downloaded first, so no feature extraction starts until
  every audio file is available on disk.  The N files are then processed
  and stored sequentially to keep peak memory usage predictable.

Steps per song:
  1. Skip if the YouTube ID is already in Qdrant.
  2. Download the audio to a temporary scratch directory (mp3).
  3. Extract song-level features + timestamped frames in one pass.
  4. Upsert the song point (31-dim) into Qdrant.
  5. Upsert all frame points (32-dim) into Qdrant.
  6. Delete the local audio file.
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from app.config import settings
from app.database.qdrant import song_exists, upsert_frames, upsert_song
from app.ingestion.downloader import VideoMeta, download_song, search_youtube
from app.ingestion.extractor import extract_all

logger = logging.getLogger(__name__)


# ───────────────────────────  Source selection  ─────────────────────────────────


async def _fetch_spotify_candidates(n: int) -> list[tuple[VideoMeta, object]]:
    """
    Pull candidate tracks from Spotify, then resolve each to a YouTube video.
    Returns list of (VideoMeta, SpotifyTrack) tuples.  Stops as soon as `n`
    un-stored candidates are assembled (or the playlist runs out).
    """
    try:
        from app.ingestion.spotify_client import (
            SPOTIFY_CATEGORIES,
            get_featured_tracks,
            get_trending_by_category,
        )
    except ImportError:
        return []

    # Pick a random category for variety
    category = random.choice(settings.SPOTIFY_CATEGORY_POOL)
    logger.info("Spotify source: category '%s'", category)

    spotify_tracks = await asyncio.to_thread(
        get_trending_by_category,
        category,
        n * 3,  # overfetch; some may already be stored
        settings.SPOTIFY_COUNTRY,
    )
    if not spotify_tracks:
        logger.warning("Category '%s' empty, falling back to featured.", category)
        spotify_tracks = await asyncio.to_thread(
            get_featured_tracks, n * 3, settings.SPOTIFY_COUNTRY
        )

    candidates: list[tuple[VideoMeta, object]] = []
    for sp_track in spotify_tracks:
        if len(candidates) >= n:
            break

        # Search YouTube for the actual audio file
        yt_results = await search_youtube(sp_track.youtube_query, max_results=3)
        if not yt_results:
            logger.debug("No YouTube result for: %s", sp_track.youtube_query)
            continue

        yt_meta = yt_results[0]

        # Prefer Spotify's title/artist over YouTube's
        enriched_meta = VideoMeta(
            id=yt_meta.id,
            title=sp_track.title,
            artist=sp_track.artist,
            duration_s=yt_meta.duration_s,
            webpage_url=yt_meta.webpage_url,
        )
        candidates.append((enriched_meta, sp_track))

    return candidates


async def _fetch_youtube_candidates(n: int) -> list[tuple[VideoMeta, None]]:
    """Fallback: search YouTube directly using the query pool."""
    query = random.choice(settings.YOUTUBE_QUERY_POOL)
    logger.info("YouTube fallback source: query '%s'", query)
    results = await search_youtube(query, max_results=n * 3)
    return [(r, None) for r in results]


# ───────────────────────────  Single-song ingestion  ──────────────────────────


async def ingest_song(
    video_id: str,
    meta: VideoMeta,
    spotify_track=None,  # SpotifyTrack | None
) -> bool:
    """
    Extract, embed, and store one song that has already been downloaded.
    `audio_path` must exist on disk; it is deleted in the finally block.
    Returns True on success, False if skipped/errored.
    Called after all batch downloads complete.
    """
    if song_exists(video_id):
        logger.info("Skip (already stored): %s – %s", meta.artist, meta.title)
        return False

    audio_path: Path | None = None
    try:
        logger.info("Downloading: %s – %s (%s)", meta.artist, meta.title, video_id)
        audio_path, refreshed_meta = await download_song(video_id, settings.SCRATCH_DIR)
        final_meta = refreshed_meta if refreshed_meta.title != "Unknown Title" else meta

        # Extract
        song_features, frame_list = await extract_all(audio_path)

        # Build spotify sub-payload
        sp_data: dict = {}
        if spotify_track is not None:
            sp_data = {
                "spotify_id": spotify_track.spotify_id,
                "spotify_url": spotify_track.spotify_url,
                "spotify_preview_url": spotify_track.preview_url,
                "album": spotify_track.album,
                "genres": spotify_track.genres,
                "popularity": spotify_track.popularity,
                "cover_url": spotify_track.cover_url,
                "spotify_audio_features": spotify_track.audio_features.to_dict(),
            }

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
            spectral_flux_mean=song_features.spectral_flux_mean,
            spectral_flux_std=song_features.spectral_flux_std,
            harmonic_ratio=song_features.harmonic_ratio,
            mfcc_means=song_features.mfcc_means,
            instrument_profile=song_features.instrument_profile,
            **sp_data,
        )

        n_frames = upsert_frames(
            song_id=song_point_id,
            youtube_id=video_id,
            title=final_meta.title,
            artist=final_meta.artist,
            frames=frame_list,
        )

        logger.info(
            "Ingested: %s – %s (%d frames)",
            final_meta.artist,
            final_meta.title,
            n_frames,
        )
        return True

    except Exception as exc:
        logger.error("Failed to ingest %s: %s", video_id, exc, exc_info=True)
        return False

    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink(missing_ok=True)


# ───────────────────────────  Ingestion cycle  ─────────────────────────────────


async def run_ingestion_cycle(n_songs: int = settings.SONGS_PER_RUN) -> dict:
    """
    One full ingestion cycle.

    1. Fetch N candidate (meta, spotify_track) pairs from Spotify or YouTube.
    2. Filter out already-stored songs.
    3. Download ALL eligible audio files to scratch dir first.
    4. Process (extract + store) them SEQUENTIALLY once all downloads are ready.
    5. Return a summary dict.
    """
    use_spotify = bool(settings.SPOTIFY_CLIENT_ID and settings.SPOTIFY_CLIENT_SECRET)

    # ── Step 1: fetch candidates ─────────────────────────────────────────────
    if use_spotify:
        candidates = await _fetch_spotify_candidates(n_songs)
    else:
        candidates = await _fetch_youtube_candidates(n_songs)

    if not candidates:
        logger.warning("No candidates found, cycle aborted.")
        return {
            "source": "spotify" if use_spotify else "youtube",
            "ingested": 0,
            "skipped": 0,
            "errors": 0,
            "downloads": 0,
        }

    # ── Step 2: filter stored + cap at n_songs ──────────────────────────────
    eligible: list[tuple[VideoMeta, object]] = []
    skipped_pre = 0
    for meta, sp_track in candidates:
        if song_exists(meta.id):
            skipped_pre += 1
            continue
        eligible.append((meta, sp_track))
        if len(eligible) >= n_songs:
            break

    if not eligible:
        logger.info("All candidates already stored (skipped=%d).", skipped_pre)
        return {
            "source": "spotify" if use_spotify else "youtube",
            "ingested": 0,
            "skipped": skipped_pre,
            "errors": 0,
            "downloads": 0,
        }

    logger.info(
        "Cycle: %d eligible song(s) to download+process (skipped %d already stored).",
        len(eligible),
        skipped_pre,
    )

    # ── Step 3: download ALL eligible songs first ────────────────────────────
    downloaded: list[tuple[Path, VideoMeta, object]] = []
    dl_errors = 0

    for meta, sp_track in eligible:
        try:
            logger.info("[Download] %s – %s", meta.artist, meta.title)
            audio_path, refreshed = await download_song(meta.id, settings.SCRATCH_DIR)
            final_meta = refreshed if refreshed.title != "Unknown Title" else meta
            downloaded.append((audio_path, final_meta, sp_track))
        except Exception as exc:
            logger.error("Download failed for %s: %s", meta.id, exc)
            dl_errors += 1

    if not downloaded:
        logger.warning("All downloads failed.")
        return {
            "source": "spotify" if use_spotify else "youtube",
            "ingested": 0,
            "skipped": skipped_pre,
            "errors": dl_errors,
            "downloads": 0,
        }

    logger.info(
        "All downloads complete (%d/%d). Starting sequential extraction.",
        len(downloaded),
        len(eligible),
    )

    # ── Step 4: process sequentially ───────────────────────────────────────
    ingested = 0
    proc_errors = 0

    for audio_path, final_meta, sp_track in downloaded:
        try:
            song_features, frame_list = await extract_all(audio_path)

            sp_data: dict = {}
            if sp_track is not None:
                sp_data = {
                    "spotify_id": sp_track.spotify_id,
                    "spotify_url": sp_track.spotify_url,
                    "spotify_preview_url": sp_track.preview_url,
                    "album": sp_track.album,
                    "genres": sp_track.genres,
                    "popularity": sp_track.popularity,
                    "cover_url": sp_track.cover_url,
                    "spotify_audio_features": sp_track.audio_features.to_dict(),
                }

            song_point_id = upsert_song(
                vector=song_features.to_embedding(),
                title=final_meta.title,
                artist=final_meta.artist,
                youtube_id=final_meta.id,
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
                spectral_flux_mean=song_features.spectral_flux_mean,
                spectral_flux_std=song_features.spectral_flux_std,
                harmonic_ratio=song_features.harmonic_ratio,
                mfcc_means=song_features.mfcc_means,
                instrument_profile=song_features.instrument_profile,
                **sp_data,
            )

            n_frames = upsert_frames(
                song_id=song_point_id,
                youtube_id=final_meta.id,
                title=final_meta.title,
                artist=final_meta.artist,
                frames=frame_list,
            )

            logger.info(
                "[%d/%d] Ingested: %s – %s (%d frames)",
                ingested + 1,
                len(downloaded),
                final_meta.artist,
                final_meta.title,
                n_frames,
            )
            ingested += 1

        except Exception as exc:
            logger.error(
                "Processing failed for %s – %s: %s",
                final_meta.artist,
                final_meta.title,
                exc,
                exc_info=True,
            )
            proc_errors += 1
        finally:
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)

    summary = {
        "source": "spotify" if use_spotify else "youtube",
        "downloads": len(downloaded),
        "ingested": ingested,
        "skipped": skipped_pre,
        "errors": dl_errors + proc_errors,
    }
    logger.info("Cycle complete: %s", summary)
    return summary
