"""
Legal audio preview client and in-memory decoder.

Pulls official 30-second previews and metadata from public APIs (iTunes Search,
Deezer, Audius) directly into RAM and decodes to mono float32 PCM via PyAV.
Zero disk I/O, zero stream-ripping, zero copyright circumvention.
"""

from __future__ import annotations

import io
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import av
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Predefined discovery queries across genres for automated rotation
DEFAULT_DISCOVERY_QUERIES = [
    "Daft Punk",
    "Afrobeats 2025",
    "Synthwave electronic",
    "Lo-fi hip hop chill",
    "Indie pop summer",
    "Deep house melodic",
    "R&B soul acoustic",
    "Pop hits",
    "Alternative rock",
    "Ambient chill piano",
]


@dataclass
class PreviewTrack:
    track_id: str
    title: str
    artist: str
    album: str
    store_url: str
    cover_url: str
    duration_s: float
    source: str
    audio_samples: np.ndarray  # mono float32 array at target sample rate


def decode_audio_in_memory(
    audio_bytes: bytes, target_sr: int = 22050
) -> tuple[np.ndarray, float]:
    """
    Decodes audio bytes (AAC, MP3, M4A, etc.) in RAM directly into a mono float32 numpy array.
    Zero disk writes, zero subprocesses.
    """
    t0 = time.perf_counter()
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)

    chunks = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray()[0])

    if not chunks:
        raise ValueError("No audio frames decoded from stream")

    audio = np.concatenate(chunks).astype(np.float32)
    decode_duration = time.perf_counter() - t0
    return audio, decode_duration


async def fetch_itunes_preview(query: str, limit: int = 1) -> list[PreviewTrack]:
    """Fetches official 30s previews from the Apple / iTunes Search API."""
    tracks: list[PreviewTrack] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.get(
            "https://itunes.apple.com/search",
            params={"term": query, "entity": "song", "limit": limit},
        )
        if resp.status_code != 200:
            logger.warning("iTunes API returned status %d", resp.status_code)
            return []

        data = resp.json()
        for item in data.get("results", []):
            preview_url = item.get("previewUrl")
            if not preview_url:
                continue

            try:
                audio_resp = await client.get(preview_url)
                if audio_resp.status_code != 200:
                    continue
                audio, _ = decode_audio_in_memory(audio_resp.content)
                track_id = f"itunes:{item.get('trackId')}"
                tracks.append(
                    PreviewTrack(
                        track_id=track_id,
                        title=item.get("trackName", "Unknown"),
                        artist=item.get("artistName", "Unknown"),
                        album=item.get("collectionName", ""),
                        store_url=item.get("trackViewUrl", ""),
                        cover_url=item.get("artworkUrl100", "").replace(
                            "100x100bb", "600x600bb"
                        ),
                        duration_s=len(audio) / 22050,
                        source="itunes",
                        audio_samples=audio,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to decode iTunes preview: %s", exc)

    return tracks


async def fetch_deezer_preview(query: str, limit: int = 1) -> list[PreviewTrack]:
    """Fetches official 30s previews from the Deezer API."""
    tracks: list[PreviewTrack] = []
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.get(
            "https://api.deezer.com/search",
            params={"q": query, "limit": limit},
        )
        if resp.status_code != 200:
            logger.warning("Deezer API returned status %d", resp.status_code)
            return []

        data = resp.json()
        for item in data.get("data", []):
            preview_url = item.get("preview")
            if not preview_url:
                continue

            try:
                audio_resp = await client.get(preview_url)
                if audio_resp.status_code != 200:
                    continue
                audio, _ = decode_audio_in_memory(audio_resp.content)
                track_id = f"deezer:{item.get('id')}"
                tracks.append(
                    PreviewTrack(
                        track_id=track_id,
                        title=item.get("title", "Unknown"),
                        artist=item.get("artist", {}).get("name", "Unknown"),
                        album=item.get("album", {}).get("title", ""),
                        store_url=item.get("link", ""),
                        cover_url=item.get("album", {}).get("cover_big", ""),
                        duration_s=len(audio) / 22050,
                        source="deezer",
                        audio_samples=audio,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to decode Deezer preview: %s", exc)

    return tracks


async def fetch_preview_candidates(
    n: int = 4, query: str | None = None
) -> list[PreviewTrack]:
    """
    Fetches candidate tracks from legal preview sources (iTunes with Deezer fallback).
    """
    search_term = query or random.choice(DEFAULT_DISCOVERY_QUERIES)
    logger.info("Fetching legal preview candidates for query: '%s'", search_term)

    # Primary: iTunes Search API
    tracks = await fetch_itunes_preview(search_term, limit=n)
    if len(tracks) < n:
        # Secondary: Deezer API
        needed = n - len(tracks)
        deezer_tracks = await fetch_deezer_preview(search_term, limit=needed)
        tracks.extend(deezer_tracks)

    return tracks
