"""
Qdrant client wrapper.

Handles connection, collection bootstrapping, upsert, and search.
Collection uses cosine distance so all embedding vectors should be
normalized to [0, 1] per-dimension (see extractor.py).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton so the connection is reused across requests/tasks.
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.QDRANT_URL:
            # ── Cloud mode (Qdrant Cloud / any hosted instance) ──────────────
            _client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            logger.info("Qdrant client connected (cloud) → %s", settings.QDRANT_URL)
        else:
            # ── Local mode (Docker Compose / bare metal) ─────────────────────
            kwargs: dict[str, Any] = {
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
            }
            if settings.QDRANT_API_KEY:
                kwargs["api_key"] = settings.QDRANT_API_KEY
            _client = QdrantClient(**kwargs)
            logger.info(
                "Qdrant client connected (local) → %s:%s",
                settings.QDRANT_HOST,
                settings.QDRANT_PORT,
            )
    return _client


def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist.

    If the collection exists but has the wrong vector dimension (e.g. after
    an embedding schema change), it is deleted and recreated.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.QDRANT_COLLECTION in existing:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        current_dim = info.config.params.vectors.size  # type: ignore[union-attr]
        if current_dim != settings.EMBEDDING_DIM:
            logger.warning(
                "Collection '%s' has dim=%d but code expects %d — recreating.",
                settings.QDRANT_COLLECTION,
                current_dim,
                settings.EMBEDDING_DIM,
            )
            client.delete_collection(settings.QDRANT_COLLECTION)
        else:
            logger.debug("Collection '%s' already exists (dim=%d).", settings.QDRANT_COLLECTION, current_dim)
            return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=qdrant_models.VectorParams(
            size=settings.EMBEDDING_DIM,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.info(
        "Created Qdrant collection '%s' (dim=%d)",
        settings.QDRANT_COLLECTION,
        settings.EMBEDDING_DIM,
    )


def upsert_song(
    *,
    vector: list[float],
    title: str,
    artist: str,
    youtube_id: str,
    duration_s: float,
    tempo_bpm: float,
    key: str,
    mode: str,
    acousticness: float,
    danceability: float,
    energy: float,
    instrumentalness: float,
    liveness: float,
    speechiness: float,
    valence: float,
    rms_mean: float,
    spectral_centroid_mean: float,
    spectral_bandwidth_mean: float,
    spectral_rolloff_mean: float,
    zcr_mean: float,
    # New spectral features
    spectral_flux_mean: float = 0.0,
    spectral_flux_std: float = 0.0,
    harmonic_ratio: float = 0.0,
    mfcc_means: list[float] | None = None,
    instrument_profile: dict | None = None,
    # Spotify fields (optional — only set when ingested via Spotify source)
    spotify_id: str | None = None,
    spotify_url: str | None = None,
    spotify_preview_url: str | None = None,
    album: str | None = None,
    genres: list[str] | None = None,
    popularity: int | None = None,
    cover_url: str | None = None,
    spotify_audio_features: dict | None = None,
) -> str:
    """Insert or update a single song point. Returns the point UUID."""
    client = get_client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"yt:{youtube_id}"))

    payload: dict[str, Any] = {
        # ── Identity ──────────────────────────────────────────────────────────
        "title": title,
        "artist": artist,
        "youtube_id": youtube_id,
        "youtube_url": f"https://www.youtube.com/watch?v={youtube_id}",
        "duration_s": duration_s,
        # ── Tempo / key ───────────────────────────────────────────────────────
        "tempo_bpm": tempo_bpm,
        "key": key,
        "mode": mode,
        # ── VibeNet features ──────────────────────────────────────────────────
        "acousticness": acousticness,
        "danceability": danceability,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "speechiness": speechiness,
        "valence": valence,
        # ── Librosa scalars ───────────────────────────────────────────────────
        "rms_mean": rms_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "zcr_mean": zcr_mean,
        # ── New spectral features ─────────────────────────────────────────────
        "spectral_flux_mean": spectral_flux_mean,
        "spectral_flux_std": spectral_flux_std,
        "harmonic_ratio": harmonic_ratio,
        "mfcc_means": mfcc_means or [],
        "instrument_profile": instrument_profile or {},
    }

    # ── Spotify sub-payload (only added when available) ───────────────────────
    if spotify_id:
        payload["spotify_id"] = spotify_id
        payload["spotify_url"] = spotify_url or ""
        payload["spotify_preview_url"] = spotify_preview_url
        payload["album"] = album or ""
        payload["genres"] = genres or []
        payload["popularity"] = popularity or 0
        payload["cover_url"] = cover_url
        payload["spotify_audio_features"] = spotify_audio_features or {}

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    logger.info("Upserted song '%s \u2013 %s' (id=%s)", artist, title, point_id)
    return point_id


def song_exists(youtube_id: str) -> bool:
    """Return True if a song with this YouTube ID is already in the collection."""
    client = get_client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"yt:{youtube_id}"))
    results = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=[point_id],
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def search_songs(
    query_vector: list[float],
    limit: int = 10,
    filters: qdrant_models.Filter | None = None,
) -> list[dict]:
    """
    Find the `limit` most similar songs to `query_vector`.
    Returns a list of payload dicts augmented with 'score' and 'id'.
    """
    client = get_client()
    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        query_filter=filters,
        with_payload=True,
    )
    results = []
    for hit in hits:
        entry = dict(hit.payload or {})
        entry["id"] = str(hit.id)
        entry["score"] = hit.score
        results.append(entry)
    return results


def collection_stats() -> dict:
    """Return basic stats about both collections."""
    client = get_client()
    songs = client.get_collection(settings.QDRANT_COLLECTION)
    try:
        frames = client.get_collection(settings.FRAME_COLLECTION)
        frame_count = frames.points_count
    except Exception:
        frame_count = None
    return {
        "total_songs": songs.points_count,
        "total_frames": frame_count,
        "status": songs.status,
        "vectors_count": songs.vectors_count,
        "indexed_vectors_count": songs.indexed_vectors_count,
    }


# ─────────────────────── Frame collection ─────────────────────────────────────


def ensure_frame_collection() -> None:
    """Create the frame-level collection, recreating on dimension mismatch."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.FRAME_COLLECTION in existing:
        info = client.get_collection(settings.FRAME_COLLECTION)
        current_dim = info.config.params.vectors.size  # type: ignore[union-attr]
        if current_dim != settings.FRAME_EMBEDDING_DIM:
            logger.warning(
                "Frame collection '%s' has dim=%d but code expects %d — recreating.",
                settings.FRAME_COLLECTION,
                current_dim,
                settings.FRAME_EMBEDDING_DIM,
            )
            client.delete_collection(settings.FRAME_COLLECTION)
        else:
            logger.debug("Frame collection '%s' already exists (dim=%d).", settings.FRAME_COLLECTION, current_dim)
            return

    client.create_collection(
        collection_name=settings.FRAME_COLLECTION,
        vectors_config=qdrant_models.VectorParams(
            size=settings.FRAME_EMBEDDING_DIM,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.info(
        "Created frame collection '%s' (dim=%d)",
        settings.FRAME_COLLECTION,
        settings.FRAME_EMBEDDING_DIM,
    )


def frames_exist(youtube_id: str) -> bool:
    """Return True if frames for this song are already stored."""
    client = get_client()
    # Search for any frame with matching youtube_id payload filter
    hits, _ = client.scroll(
        collection_name=settings.FRAME_COLLECTION,
        scroll_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="youtube_id",
                    match=qdrant_models.MatchValue(value=youtube_id),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(hits) > 0


def upsert_frames(
    *,
    song_id: str,  # UUID of the parent song point
    youtube_id: str,
    title: str,
    artist: str,
    frames: list,  # list[FrameFeatures] from extractor
) -> int:
    """
    Batch-upsert all frame windows for a song.
    Each frame point is keyed by a deterministic UUID derived from
    (youtube_id, frame_index) so re-ingestion is idempotent.
    Returns the number of frames stored.
    """
    client = get_client()
    points: list[qdrant_models.PointStruct] = []

    for i, frame in enumerate(frames):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"yt:{youtube_id}:frame:{i}"))
        embedding = frame.to_embedding()
        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "song_id": song_id,
                    "youtube_id": youtube_id,
                    "title": title,
                    "artist": artist,
                    "frame_index": i,
                    "timestamp_start": frame.timestamp_start,
                    "timestamp_end": frame.timestamp_end,
                    # Raw feature values for inspection
                    "mfcc_means": frame.mfcc_means,
                    "chroma_means": frame.chroma_means,
                    "rms_mean": frame.rms_mean,
                    "zcr_mean": frame.zcr_mean,
                    "spectral_centroid_mean": frame.spectral_centroid_mean,
                    "spectral_bandwidth_mean": frame.spectral_bandwidth_mean,
                    "spectral_rolloff_mean": frame.spectral_rolloff_mean,
                },
            )
        )

    if points:
        # Qdrant recommends batches of ≤100 for large payloads
        batch_size = 100
        for batch_start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=settings.FRAME_COLLECTION,
                points=points[batch_start : batch_start + batch_size],
            )
        logger.info("Upserted %d frames for '%s – %s'", len(points), artist, title)
    return len(points)


def search_frames(
    query_vector: list[float],
    limit: int = 10,
) -> list[dict]:
    """
    Find the `limit` frames most similar to `query_vector`.
    Returns payload dicts enriched with 'score' and 'id'.
    """
    client = get_client()
    hits = client.search(
        collection_name=settings.FRAME_COLLECTION,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
    results = []
    for hit in hits:
        entry = dict(hit.payload or {})
        entry["id"] = str(hit.id)
        entry["score"] = hit.score
        results.append(entry)
    return results
