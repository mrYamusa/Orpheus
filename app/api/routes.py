"""
FastAPI route definitions.

Endpoints:
  GET  /health                   — liveness check
  GET  /stats                    — Qdrant collection stats
  GET  /scheduler/status         — next scheduled run info
  POST /scheduler/trigger        — manually trigger an ingestion cycle now
  POST /ingest/song              — ingest a specific YouTube video ID
  POST /search                   — semantic song search from a JSON query
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.database.qdrant import (
    collection_stats,
    search_frames,
    search_songs,
    song_exists,
)
from app.ingestion.downloader import search_youtube
from app.ingestion.extractor import extract_snippet_embedding
from app.ingestion.pipeline import ingest_song, run_ingestion_cycle
from app.scheduler.jobs import get_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────── Health ───────────────────────────


@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


# ─────────────────────────── Stats ────────────────────────────


@router.get("/stats", tags=["Library"])
async def stats():
    """Return high-level Qdrant collection statistics."""
    try:
        return collection_stats()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Qdrant unavailable: {exc}"
        ) from exc


# ──────────────────────── Scheduler ───────────────────────────


@router.get("/scheduler/status", tags=["Scheduler"])
async def scheduler_status():
    """Return the next scheduled job fire time."""
    sched = get_scheduler()
    job = sched.get_job("ingestion_job")
    if not job:
        return {"running": False, "next_run": None}
    return {
        "running": sched.running,
        "next_run": str(job.next_run_time),
        "schedule": f"every {settings.SCHEDULE_HOURS} hour(s)",
        "songs_per_run": settings.SONGS_PER_RUN,
    }


@router.post("/scheduler/trigger", tags=["Scheduler"])
async def trigger_now(background_tasks: BackgroundTasks):
    """Manually fire an ingestion cycle in the background right now."""
    background_tasks.add_task(run_ingestion_cycle)
    return {"message": "Ingestion cycle queued."}


# ──────────────────────── Ingest ──────────────────────────────


class IngestRequest(BaseModel):
    youtube_id: str = Field(..., description="YouTube video ID (11-char string)")


@router.post("/ingest/song", tags=["Ingestion"])
async def ingest_single(req: IngestRequest, background_tasks: BackgroundTasks):
    """Queue a specific YouTube video for ingestion."""
    if song_exists(req.youtube_id):
        return {"message": "Already in library.", "youtube_id": req.youtube_id}

    # Fetch metadata first to validate the ID
    candidates = await search_youtube(
        f"https://www.youtube.com/watch?v={req.youtube_id}", max_results=1
    )
    if not candidates:
        raise HTTPException(
            status_code=404, detail="YouTube video not found or unavailable."
        )

    meta = candidates[0]
    background_tasks.add_task(ingest_song, req.youtube_id, meta)
    return {
        "message": "Ingestion queued.",
        "youtube_id": req.youtube_id,
        "title": meta.title,
        "artist": meta.artist,
    }


# ──────────────────────── Search ──────────────────────────────


class SearchRequest(BaseModel):
    """
    Structured query produced by the LLM after transcribing user voice.

    All float fields are in [0, 1] unless noted.  Omit any field to
    leave it unconstrained.
    """

    # ── Vibe targets (used to build the query vector) ──
    energy: float | None = Field(None, ge=0, le=1)
    valence: float | None = Field(None, ge=0, le=1)
    danceability: float | None = Field(None, ge=0, le=1)
    acousticness: float | None = Field(None, ge=0, le=1)
    instrumentalness: float | None = Field(None, ge=0, le=1)
    speechiness: float | None = Field(None, ge=0, le=1)
    liveness: float | None = Field(None, ge=0, le=1)

    # ── Hard filters (Qdrant payload filters) ──
    tempo_min: float | None = Field(None, description="Minimum BPM")
    tempo_max: float | None = Field(None, description="Maximum BPM")
    mode: str | None = Field(None, description="'major' or 'minor'")
    energy_min: float | None = Field(None, ge=0, le=1)
    valence_min: float | None = Field(None, ge=0, le=1)
    valence_max: float | None = Field(None, ge=0, le=1)
    danceability_min: float | None = Field(None, ge=0, le=1)

    # ── Result control ──
    limit: int = Field(10, ge=1, le=50)


def _build_query_vector(req: SearchRequest) -> list[float]:
    """
    Convert the SearchRequest into a 28-dim query vector.
    Missing vibe values default to 0.5 (neutral / "don't care").
    Librosa dimensions default to neutral midpoints.
    """
    neutral = 0.5

    vibe_dims = [
        req.acousticness if req.acousticness is not None else neutral,
        req.danceability if req.danceability is not None else neutral,
        req.energy if req.energy is not None else neutral,
        req.instrumentalness if req.instrumentalness is not None else neutral,
        req.liveness if req.liveness is not None else neutral,
        req.speechiness if req.speechiness is not None else neutral,
        req.valence if req.valence is not None else neutral,
    ]

    # For librosa dims we can't know user intent so use neutral midpoints
    librosa_dims = [neutral] * 8  # tempo, key, mode, rms, centroid, bw, rolloff, zcr

    # For MFCCs also neutral (0.5 after tanh squash means raw value ≈ 0)
    mfcc_dims = [neutral] * 13

    return vibe_dims + librosa_dims + mfcc_dims


def _build_filters(req: SearchRequest) -> qdrant_models.Filter | None:
    must: list = []

    def range_cond(key: str, gte=None, lte=None):
        rng = {}
        if gte is not None:
            rng["gte"] = gte
        if lte is not None:
            rng["lte"] = lte
        if rng:
            must.append(
                qdrant_models.FieldCondition(key=key, range=qdrant_models.Range(**rng))
            )

    range_cond("tempo_bpm", gte=req.tempo_min, lte=req.tempo_max)
    range_cond("energy", gte=req.energy_min)
    range_cond("valence", gte=req.valence_min, lte=req.valence_max)
    range_cond("danceability", gte=req.danceability_min)

    if req.mode:
        must.append(
            qdrant_models.FieldCondition(
                key="mode",
                match=qdrant_models.MatchValue(value=req.mode),
            )
        )

    return qdrant_models.Filter(must=must) if must else None


@router.post("/search", tags=["Search"])
async def search(req: SearchRequest):
    """
    Semantic song search.

    Send a structured JSON produced by your LLM (from voice transcription)
    and get back a ranked playlist of matching songs from the library.
    """
    query_vector = _build_query_vector(req)
    filters = _build_filters(req)

    try:
        results = search_songs(query_vector, limit=req.limit, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}") from exc

    return {
        "count": len(results),
        "playlist": results,
    }


# ──────────────────────── Snippet match ───────────────────────────────────────


@router.post("/match/snippet", tags=["Search"])
async def match_snippet(
    file: UploadFile = File(
        ...,
        description="Short audio clip (mp3/wav/m4a/ogg). Typically 3–30 seconds.",
    ),
    similar_limit: int = 5,
):
    """
    **Shazam-style snippet matching.**

    Upload a short recording (a song playing in the background, a hummed melody,
    a voice memo) and get back:
    - `match` — the most likely song and the timestamp window it was heard at.
    - `similar_songs` — other songs in the library that sound like it.

    Supported formats: mp3, wav, m4a, ogg, flac (anything ffmpeg can decode).
    The audio file is deleted immediately after extraction.
    """
    # Save upload to scratch dir with random filename
    settings.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "snippet.mp3").suffix or ".mp3"
    tmp_path = settings.SCRATCH_DIR / f"snippet_{uuid.uuid4().hex}{suffix}"

    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)

        # Extract 30-dim snippet embedding
        try:
            snippet_vec = await extract_snippet_embedding(tmp_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"Could not process audio: {exc}"
            ) from exc

        # Search frame collection for the closest matching window
        try:
            frame_hits = search_frames(snippet_vec, limit=similar_limit * 3)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Frame search failed: {exc}"
            ) from exc

        if not frame_hits:
            return {"match": None, "similar_songs": []}

        # De-duplicate by song: keep best-scoring frame per song
        seen: dict[str, dict] = {}
        for hit in frame_hits:
            yid = hit.get("youtube_id", "")
            if yid not in seen or hit["score"] > seen[yid]["score"]:
                seen[yid] = hit

        ranked = sorted(seen.values(), key=lambda h: h["score"], reverse=True)
        best = ranked[0]

        match_result = {
            "title": best.get("title"),
            "artist": best.get("artist"),
            "youtube_id": best.get("youtube_id"),
            "heard_at": {
                "start_s": best.get("timestamp_start"),
                "end_s": best.get("timestamp_end"),
            },
            "confidence": round(best["score"], 4),
        }

        # Use the best frame's song_id to do a song-level vibe search for similar tracks
        # We re-search the SONG collection using the snippet vector projected to 28-dims.
        # Since dims 0-14 overlap in meaning (MFCCs → dims 15-27 in song, 0-12 in frame)
        # we build a neutral song vector and use payload filters instead.
        similar: list[dict] = []
        if len(ranked) > 1:
            # Other songs found in frame search are already "similar" by acoustic features
            similar = [
                {
                    "title": r.get("title"),
                    "artist": r.get("artist"),
                    "youtube_id": r.get("youtube_id"),
                    "score": round(r["score"], 4),
                }
                for r in ranked[1 : similar_limit + 1]
            ]

        return {
            "match": match_result,
            "similar_songs": similar,
        }

    finally:
        tmp_path.unlink(missing_ok=True)
