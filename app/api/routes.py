"""
FastAPI route definitions (Modernized Legal Architecture).

Endpoints:
  GET  /health                   — liveness check
  GET  /stats                    — Qdrant and fingerprint collection statistics
  GET  /scheduler/status         — next scheduled ingestion run info
  POST /scheduler/trigger        — manually trigger an ingestion cycle now
  POST /ingest/song              — ingest a specific track via search/preview
  POST /search                   — semantic natural language and vibe search
  POST /match/snippet            — Shazam-style instant audio snippet recognition
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.database.fingerprint import get_fingerprinter
from app.database.qdrant import (
    collection_stats,
    search_songs,
    song_exists,
)
from app.ingestion.extractor import extract_text_embedding
from app.ingestion.pipeline import ingest_song, run_ingestion_cycle
from app.ingestion.preview_client import decode_audio_in_memory
from app.scheduler.jobs import get_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────── Health ───────────────────────────
@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "architecture": "legal_preview_v2"}


# ─────────────────────────── Stats ────────────────────────────
@router.get("/stats", tags=["Library"])
async def stats():
    """Return high-level Qdrant collection & fingerprint statistics."""
    try:
        q_stats = collection_stats()
    except Exception:
        q_stats = {"status": "offline", "total_songs": 0}

    # Query fingerprint count
    fp = get_fingerprinter()
    with fp._get_conn() as conn:
        song_count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        hash_count = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]

    return {
        "qdrant": q_stats,
        "fingerprints": {
            "indexed_songs": song_count,
            "total_hashes": hash_count,
        },
    }


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
        "schedule": f"every {settings.SCHEDULE_MINUTES} minute(s)",
        "songs_per_run": settings.SONGS_PER_RUN,
        "source": "legal_previews (Apple / Deezer / Audius)",
    }


@router.post("/scheduler/trigger", tags=["Scheduler"])
async def trigger_now(background_tasks: BackgroundTasks):
    """Manually fire an ingestion cycle in the background right now."""
    background_tasks.add_task(run_ingestion_cycle)
    return {"message": "Legal preview ingestion cycle queued in background."}


# ──────────────────────── Ingest ──────────────────────────────
class IngestRequest(BaseModel):
    query: str = Field(
        ...,
        description="Song search term or artist name (e.g. 'Daft Punk One More Time')",
    )


@router.post("/ingest/song", tags=["Ingestion"])
async def ingest_single(req: IngestRequest, background_tasks: BackgroundTasks):
    """Queue a specific track for legal preview ingestion."""
    background_tasks.add_task(ingest_song, req.query)
    return {
        "message": "Ingestion queued via legal preview stream.",
        "query": req.query,
    }


# ──────────────────────── Search ──────────────────────────────
class SearchRequest(BaseModel):
    """
    Search query supporting both natural language prompts and vibe filters.
    """

    # ── Natural Language Voice / Text Prompt ──
    prompt: str | None = Field(
        None,
        description="Natural language search prompt (e.g. 'driving late at night with dark synthwave')",
    )

    # ── Vibe targets (optional manual overrides) ──
    energy: float | None = Field(None, ge=0, le=1)
    valence: float | None = Field(None, ge=0, le=1)
    danceability: float | None = Field(None, ge=0, le=1)
    acousticness: float | None = Field(None, ge=0, le=1)
    instrumentalness: float | None = Field(None, ge=0, le=1)

    # ── Hard filters ──
    tempo_min: float | None = Field(None, description="Minimum BPM")
    tempo_max: float | None = Field(None, description="Maximum BPM")
    mode: str | None = Field(None, description="'major' or 'minor'")

    # ── Result control ──
    limit: int = Field(10, ge=1, le=50)


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
    range_cond("energy", gte=req.energy)
    range_cond("valence", gte=req.valence)
    range_cond("danceability", gte=req.danceability)

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
    Semantic song search via natural language prompt and/or vibe criteria.
    """
    # Generate 512-dim query vector from prompt or fallback vibe
    if req.prompt:
        query_vector = await extract_text_embedding(req.prompt)
    else:
        # Default prompt synthesized from vibe values
        synthetic_prompt = f"music with energy {req.energy or 0.5} and valence {req.valence or 0.5}"
        query_vector = await extract_text_embedding(synthetic_prompt)

    filters = _build_filters(req)

    try:
        results = search_songs(query_vector, limit=req.limit, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}") from exc

    playlist = []
    for rank, hit in enumerate(results, start=1):
        playlist.append(
            {
                "rank": rank,
                "score": round(hit.get("score", 0), 4),
                "title": hit.get("title"),
                "artist": hit.get("artist"),
                "album": hit.get("album"),
                "source": hit.get("source"),
                "duration_s": hit.get("duration_s"),
                "links": {
                    "store": hit.get("store_url"),
                    "youtube": hit.get("youtube_url"),
                },
                "cover_url": hit.get("cover_url"),
            }
        )

    return {
        "count": len(playlist),
        "prompt": req.prompt,
        "playlist": playlist,
    }


# ──────────────────────── Snippet match ───────────────────────────────────────
@router.post("/match/snippet", tags=["Search"])
async def match_snippet(
    file: UploadFile = File(
        ...,
        description="Audio snippet recording (mp3/wav/m4a/ogg). Decoded 100% in memory.",
    ),
):
    """
    **Instant Shazam-Style Snippet Matching (Acoustic Constellation Fingerprint).**

    Upload a short recording (humming, microphone audio, ambient clip) and get back:
    - `match` — Exact song, artist, album, store link, and timestamp where it was heard.
    - `confidence_votes` — Number of verified matching spectral peak pairs.
    - `query_time_ms` — Sub-100ms match latency.
    """
    try:
        contents = await file.read()
        audio, decode_time = decode_audio_in_memory(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not decode audio stream: {exc}"
        ) from exc

    fingerprinter = get_fingerprinter()
    match_result = fingerprinter.match_snippet(audio)
    match_result["decode_time_ms"] = round(decode_time * 1000, 2)

    return match_result
