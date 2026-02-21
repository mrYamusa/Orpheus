"""
Spotify integration via Spotipy (Client Credentials flow — no user login).

Pulls tracks via Spotify Recommendations (seed-by-genre) — the only stable
discovery endpoint after Spotify removed Browse/Categories and
Featured-Playlists in November 2024.  audio-features was also deprecated at
that time; we attempt it but silently no-op on 403/404 so the pipeline never
crashes.

Public API
──────────
  get_trending_by_category(category_id, limit)  → list[SpotifyTrack]
  get_featured_tracks(limit)                    → list[SpotifyTrack]
  SPOTIFY_CATEGORIES                            → dict of category slugs → label

SpotifyTrack carries enough info to:
  1. Search YouTube for the audio: track.youtube_query
  2. Store a Spotify deep-link in the Qdrant payload: track.spotify_url
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from app.config import settings

logger = logging.getLogger(__name__)

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─────────────────────────── Data classes ─────────────────────────────────────


@dataclass
class SpotifyAudioFeatures:
    """Subset of Spotify's audio-features object, stored as a sub-payload."""

    tempo: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0
    valence: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    loudness: float = 0.0  # dB (typically −60 to 0)
    key: int = 0  # 0 = C … 11 = B
    mode: int = 1  # 1 = major, 0 = minor
    time_signature: int = 4

    def key_name(self) -> str:
        return _KEY_NAMES[self.key % 12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo": self.tempo,
            "energy": self.energy,
            "danceability": self.danceability,
            "valence": self.valence,
            "acousticness": self.acousticness,
            "instrumentalness": self.instrumentalness,
            "liveness": self.liveness,
            "speechiness": self.speechiness,
            "loudness": self.loudness,
            "key": self.key_name(),
            "mode": "major" if self.mode == 1 else "minor",
            "time_signature": self.time_signature,
        }


@dataclass
class SpotifyTrack:
    """All Spotify data for one track, ready to feed into the ingestion pipeline."""

    spotify_id: str
    title: str
    artist: str  # primary artist name
    artists: list[str] = field(default_factory=list)
    album: str = ""
    genres: list[str] = field(default_factory=list)
    popularity: int = 0
    spotify_url: str = ""  # https://open.spotify.com/track/…
    preview_url: str | None = None  # 30-s Spotify preview (may be None)
    duration_ms: int = 0
    audio_features: SpotifyAudioFeatures = field(default_factory=SpotifyAudioFeatures)
    cover_url: str | None = None  # album art (640×640 or largest)

    @property
    def youtube_query(self) -> str:
        """Search term for locating the track on YouTube."""
        return f"{self.artist} {self.title} official audio"

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


# ──────────────────────── Spotify client singleton ────────────────────────────


@lru_cache(maxsize=1)
def _get_client() -> spotipy.Spotify:
    """
    Return a cached Spotipy client.
    Raises RuntimeError if credentials are missing from .env.
    """
    if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env. "
            "Get them at https://developer.spotify.com/dashboard"
        )
    auth_manager = SpotifyClientCredentials(
        client_id=settings.SPOTIFY_CLIENT_ID,
        client_secret=settings.SPOTIFY_CLIENT_SECRET,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


# ──────────────────────── Internal helpers ────────────────────────────────────


def _parse_audio_features(raw: dict | None) -> SpotifyAudioFeatures:
    if not raw:
        return SpotifyAudioFeatures()
    return SpotifyAudioFeatures(
        tempo=float(raw.get("tempo") or 0),
        energy=float(raw.get("energy") or 0),
        danceability=float(raw.get("danceability") or 0),
        valence=float(raw.get("valence") or 0),
        acousticness=float(raw.get("acousticness") or 0),
        instrumentalness=float(raw.get("instrumentalness") or 0),
        liveness=float(raw.get("liveness") or 0),
        speechiness=float(raw.get("speechiness") or 0),
        loudness=float(raw.get("loudness") or 0),
        key=int(raw.get("key") or 0),
        mode=int(raw.get("mode") or 1),
        time_signature=int(raw.get("time_signature") or 4),
    )


def _parse_track_item(item: dict) -> SpotifyTrack | None:
    """Parse a Spotify playlist-item dict into a SpotifyTrack."""
    track = item.get("track") or item  # handle playlist-item wrapper
    if not track or not track.get("id"):
        return None

    artists = [a["name"] for a in (track.get("artists") or [])]
    album_obj = track.get("album") or {}
    images = album_obj.get("images") or []
    # Prefer the largest image (usually index 0)
    cover = images[0]["url"] if images else None

    return SpotifyTrack(
        spotify_id=track["id"],
        title=track.get("name", "Unknown Title"),
        artist=artists[0] if artists else "Unknown Artist",
        artists=artists,
        album=album_obj.get("name", ""),
        genres=[],
        popularity=int(track.get("popularity") or 0),
        spotify_url=track.get("external_urls", {}).get("spotify", ""),
        preview_url=track.get("preview_url"),
        duration_ms=int(track.get("duration_ms") or 0),
        cover_url=cover,
    )


def _enrich_with_audio_features(
    sp: spotipy.Spotify,
    tracks: list[SpotifyTrack],
) -> None:
    """
    Fetch Spotify audio features in batches of 100 and attach to tracks.

    ``GET /audio-features`` was deprecated November 27 2024.  This function
    silently no-ops on 403/404 so the pipeline never crashes — SpotifyTrack
    will simply carry the default zero-valued SpotifyAudioFeatures.
    """
    ids = [t.spotify_id for t in tracks]
    for i in range(0, len(ids), 100):
        batch_ids = ids[i : i + 100]
        try:
            raw_list = sp.audio_features(batch_ids) or []
            feat_map: dict[str, dict] = {f["id"]: f for f in raw_list if f}
        except Exception as exc:
            # 403 = deprecated, 404 = gone — either way skip silently
            logger.debug("Spotify audio_features unavailable (skipping): %s", exc)
            return
        for track in tracks[i : i + 100]:
            track.audio_features = _parse_audio_features(feat_map.get(track.spotify_id))


# ──────────────────────── Category catalogue ──────────────────────────────────

# Human-readable labels for each category slug used in config.SPOTIFY_CATEGORY_POOL.
SPOTIFY_CATEGORIES: dict[str, str] = {
    "hiphop":    "Hip-Hop",
    "pop":       "Pop",
    "rnb":       "R&B",
    "latin":     "Latin",
    "edm_dance": "Electronic / Dance",
    "rock":      "Rock",
    "afro":      "Afrobeats",
    "indie_alt": "Indie / Alternative",
    "soul":      "Soul",
    "workout":   "Workout",
    "party":     "Party",
    "chill":     "Chill",
    "romance":   "Romance",
    "mood":      "Mood",
}

# Map our category slugs to valid Spotify Recommendations seed_genres.
# See: GET /recommendations/available-genre-seeds
_GENRE_SEEDS: dict[str, list[str]] = {
    "hiphop":    ["hip-hop"],
    "pop":       ["pop"],
    "rnb":       ["r-n-b"],
    "latin":     ["latin"],
    "edm_dance": ["edm", "dance"],
    "rock":      ["rock"],
    "afro":      ["afrobeat"],
    "indie_alt": ["indie", "alternative"],
    "soul":      ["soul"],
    "workout":   ["work-out"],
    "party":     ["party"],
    "chill":     ["chill"],
    "romance":   ["romance"],
    "mood":      ["pop", "soul"],  # no direct 'mood' seed — use broad mix
}

# Pool used when a category has no specific seed or for the featured fallback.
_FEATURED_SEEDS = ["pop", "hip-hop", "r-n-b", "latin", "rock"]


# ──────────────────────── Public API ──────────────────────────────────────────


def get_trending_by_category(
    category_id: str,
    limit: int = 20,
    country: str = "US",
) -> list[SpotifyTrack]:
    """
    Return trending tracks for a genre category using Spotify Recommendations.

    Uses ``sp.recommendations(seed_genres=[...], limit=limit*2)`` which is the
    stable endpoint after Spotify removed Browse/Categories in November 2024.
    Falls back to an empty list on any API error (caller handles the fallback).
    """
    sp = _get_client()
    seeds = _GENRE_SEEDS.get(category_id, ["pop"])

    # ── 1. Fetch recommendations ───────────────────────────────────────────────
    try:
        resp = sp.recommendations(
            seed_genres=seeds[:5],      # API max 5 seeds total
            limit=min(limit * 2, 100),
            country=country,
        )
        raw_tracks = (resp or {}).get("tracks") or []
    except Exception as exc:
        logger.error(
            "Spotify recommendations failed for category '%s' (seeds=%s): %s",
            category_id, seeds, exc,
        )
        return []

    # ── 2. Parse ───────────────────────────────────────────────────────────────
    tracks: list[SpotifyTrack] = []
    for item in raw_tracks:
        t = _parse_track_item(item)  # recommendations items are plain track dicts
        if t and t.duration_ms > 0:
            t.genres = [SPOTIFY_CATEGORIES.get(category_id, category_id)]
            tracks.append(t)
        if len(tracks) >= limit:
            break

    if not tracks:
        logger.warning("No tracks returned for Spotify category '%s'", category_id)
        return []

    # ── 3. Optionally enrich with audio features (best-effort) ─────────────────
    _enrich_with_audio_features(sp, tracks)

    logger.info(
        "Spotify recommendations: %d tracks for category '%s' (seeds=%s)",
        len(tracks), category_id, seeds,
    )
    return tracks


def get_featured_tracks(limit: int = 20, country: str = "US") -> list[SpotifyTrack]:
    """
    Pull a diverse cross-genre set of tracks via Spotify Recommendations.
    Used as a fallback when SPOTIFY_CATEGORY_POOL is empty or a category call fails.
    """
    sp = _get_client()
    # Pick 2 random seeds from the featured pool for variety
    seeds = random.sample(_FEATURED_SEEDS, k=min(2, len(_FEATURED_SEEDS)))

    try:
        resp = sp.recommendations(
            seed_genres=seeds,
            limit=min(limit * 2, 100),
            country=country,
        )
        raw_tracks = (resp or {}).get("tracks") or []
    except Exception as exc:
        logger.error("Spotify featured recommendations failed: %s", exc)
        return []

    tracks: list[SpotifyTrack] = []
    for item in raw_tracks:
        t = _parse_track_item(item)
        if t and t.duration_ms > 0:
            t.genres = ["featured"]
            tracks.append(t)
        if len(tracks) >= limit:
            break

    if not tracks:
        return []

    _enrich_with_audio_features(sp, tracks)
    logger.info("Spotify featured: fetched %d tracks (seeds=%s)", len(tracks), seeds)
    return tracks
