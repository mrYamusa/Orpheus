"""
Spotify integration via Spotipy (Client Credentials flow — no user login).

Discovery strategy
──────────────────
Uses ``GET /search`` (type=track, q='genre:<term>') which is the only
discovery endpoint available to new Spotify apps (created after Nov 27 2024).
A random result-page offset is used so each run surfaces different songs.

Audio-features (``GET /audio-features``) was deprecated Nov 27 2024; we
attempt it but silently no-op on 403/404 so the pipeline never crashes.

Public API
──────────
  get_trending_by_category(category_id, limit)  → list[SpotifyTrack]
  get_featured_tracks(limit)                    → list[SpotifyTrack]
  SPOTIFY_CATEGORIES                            → dict of category slugs → label
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
    """Parse a Spotify track dict (search result or playlist-item wrapper)."""
    track = item.get("track") or item  # handle playlist-item wrapper
    if not track or not track.get("id"):
        return None

    artists = [a["name"] for a in (track.get("artists") or [])]
    album_obj = track.get("album") or {}
    images = album_obj.get("images") or []
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
    Silently no-ops on 403/404 (endpoint deprecated Nov 27 2024).
    """
    ids = [t.spotify_id for t in tracks]
    for i in range(0, len(ids), 100):
        batch_ids = ids[i : i + 100]
        try:
            raw_list = sp.audio_features(batch_ids) or []
            feat_map: dict[str, dict] = {f["id"]: f for f in raw_list if f}
        except Exception as exc:
            logger.debug("Spotify audio_features unavailable (skipping): %s", exc)
            return
        for track in tracks[i : i + 100]:
            track.audio_features = _parse_audio_features(feat_map.get(track.spotify_id))


def _search_tracks(
    sp: spotipy.Spotify,
    query: str,
    limit: int,
    genre_label: str,
    market: str = "US",
) -> list[SpotifyTrack]:
    """
    Run one ``sp.search`` call with a random page offset for variety and
    return parsed SpotifyTrack objects.  Returns [] on any error.

    Spotify genre-search paging is capped much lower than the documented 1000;
    keep offset ≤ 200 to stay inside the safe range.
    """
    safe_limit = min(limit, 50)
    # Keep offset low — genre queries have a much smaller index than general search
    max_offset = max(0, min(200, 200 - safe_limit))
    offset = random.randint(0, max_offset)
    try:
        resp = sp.search(
            q=query,
            type="track",
            limit=safe_limit,
            offset=offset,
            market=market,   # avoids sending market=None which causes 400s
        )
        raw_items = ((resp or {}).get("tracks") or {}).get("items") or []
    except Exception as exc:
        logger.error("Spotify search failed (q=%r): %s", query, exc)
        return []

    tracks: list[SpotifyTrack] = []
    for item in raw_items:
        t = _parse_track_item(item)
        if t and t.duration_ms > 0:
            t.genres = [genre_label]
            tracks.append(t)
        if len(tracks) >= limit:
            break
    return tracks


# ──────────────────────── Category catalogue ──────────────────────────────────

SPOTIFY_CATEGORIES: dict[str, str] = {
    "hiphop": "Hip-Hop",
    "pop": "Pop",
    "rnb": "R&B",
    "latin": "Latin",
    "edm_dance": "Electronic / Dance",
    "rock": "Rock",
    "afro": "Afrobeats",
    "indie_alt": "Indie / Alternative",
    "soul": "Soul",
    "workout": "Workout",
    "party": "Party",
    "chill": "Chill",
    "romance": "Romance",
    "mood": "Mood",
}

# Spotify search genre terms for each category slug.
# These are passed as q='genre:"<term>"' to GET /search.
_GENRE_QUERIES: dict[str, str] = {
    "hiphop": 'genre:"hip-hop"',
    "pop": 'genre:"pop"',
    "rnb": 'genre:"r&b"',
    "latin": 'genre:"latin"',
    "edm_dance": 'genre:"edm"',
    "rock": 'genre:"rock"',
    "afro": 'genre:"afrobeats"',
    "indie_alt": 'genre:"indie"',
    "soul": 'genre:"soul"',
    "workout": 'genre:"pop" tag:new',
    "party": 'genre:"dance pop"',
    "chill": 'genre:"chill"',
    "romance": 'genre:"romance"',
    "mood": 'genre:"pop" tag:new',
}

# Cross-genre queries used by get_featured_tracks()
_FEATURED_QUERIES = [
    'genre:"pop" tag:new',
    'genre:"hip-hop" tag:new',
    'genre:"r&b" tag:new',
    'genre:"latin" tag:new',
    'genre:"afrobeats" tag:new',
]


# ──────────────────────── Public API ──────────────────────────────────────────


def get_trending_by_category(
    category_id: str,
    limit: int = 20,
    country: str = "US",
) -> list[SpotifyTrack]:
    """
    Return tracks for a genre category via Spotify Search.

    Uses ``sp.search(q='genre:"<term>"', type='track')`` with a random offset
    so consecutive runs return different results.  Falls back to an empty list
    on any API error (caller handles the fallback).

    ``country`` is accepted for API-signature compatibility but Search does not
    take a market filter — Spotify returns globally available tracks.
    """
    sp = _get_client()
    query = _GENRE_QUERIES.get(category_id, f'genre:"{category_id}"')
    label = SPOTIFY_CATEGORIES.get(category_id, category_id)

    tracks = _search_tracks(sp, query, limit, label, market=country)
    if not tracks:
        logger.warning("No tracks returned for Spotify category '%s'", category_id)
        return []

    _enrich_with_audio_features(sp, tracks)
    logger.info(
        "Spotify search: %d tracks for category '%s' (q=%r)",
        len(tracks),
        category_id,
        query,
    )
    return tracks


def get_featured_tracks(limit: int = 20, country: str = "US") -> list[SpotifyTrack]:
    """
    Pull a diverse cross-genre set of tracks via Spotify Search.
    Used as fallback when a category call returns empty.
    """
    sp = _get_client()
    query = random.choice(_FEATURED_QUERIES)

    tracks = _search_tracks(sp, query, limit, "featured", market=country)
    if not tracks:
        return []

    _enrich_with_audio_features(sp, tracks)
    logger.info("Spotify featured: %d tracks (q=%r)", len(tracks), query)
    return tracks

