"""
Spotify integration via Spotipy (Client Credentials flow — no user login).

Pulls trending tracks from Spotify categories / featured playlists and
enriches them with Spotify audio features so they can be stored alongside
the locally-extracted librosa + VibeNet features in Qdrant.

Public API
──────────
  get_trending_by_category(category_id, limit)  → list[SpotifyTrack]
  get_featured_tracks(limit)                    → list[SpotifyTrack]
  SPOTIFY_CATEGORIES                            → dict of category slugs

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
    """Fetch Spotify audio features in batches of 100 and attach to tracks."""
    ids = [t.spotify_id for t in tracks]
    for i in range(0, len(ids), 100):
        batch_ids = ids[i : i + 100]
        try:
            raw_list = sp.audio_features(batch_ids) or []
            feat_map: dict[str, dict] = {f["id"]: f for f in raw_list if f}
        except Exception as exc:
            logger.warning("Spotify audio_features batch failed: %s", exc)
            feat_map = {}
        for track in tracks[i : i + 100]:
            track.audio_features = _parse_audio_features(feat_map.get(track.spotify_id))


# ──────────────────────── Category catalogue ──────────────────────────────────

# Spotify Browse Category IDs → human label.
# Use the ID as the key when calling get_trending_by_category().
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


# ──────────────────────── Public API ──────────────────────────────────────────


def get_trending_by_category(
    category_id: str,
    limit: int = 20,
    country: str = "US",
) -> list[SpotifyTrack]:
    """
    Return trending tracks from a Spotify browse category.

    Picks one playlist at random from the category's top-3 playlists for
    variety, then pulls `limit` tracks and enriches them with audio features.
    Falls back to an empty list on any API error (caller handles the fallback).
    """
    sp = _get_client()

    # ── 1. Get playlists for the category ─────────────────────────────────────
    try:
        resp = sp.category_playlists(category_id=category_id, country=country, limit=3)
        playlist_items = (resp.get("playlists") or {}).get("items") or []
        if not playlist_items:
            logger.warning("No playlists for Spotify category '%s'", category_id)
            return []
        playlist = random.choice(playlist_items)
    except Exception as exc:
        logger.error("Spotify category_playlists failed for '%s': %s", category_id, exc)
        return []

    # ── 2. Pull tracks from the playlist ──────────────────────────────────────
    try:
        result = sp.playlist_tracks(
            playlist["id"],
            fields=(
                "items(track(id,name,artists,album,external_urls,"
                "preview_url,popularity,duration_ms))"
            ),
            limit=min(limit * 2, 50),
        )
        raw_items = (result or {}).get("items") or []
    except Exception as exc:
        logger.error("Spotify playlist_tracks failed: %s", exc)
        return []

    # ── 3. Parse (skip None / local-file entries) ──────────────────────────────
    tracks: list[SpotifyTrack] = []
    for item in raw_items:
        t = _parse_track_item(item)
        if t and t.duration_ms > 0:
            t.genres = [SPOTIFY_CATEGORIES.get(category_id, category_id)]
            tracks.append(t)
        if len(tracks) >= limit:
            break

    if not tracks:
        return []

    # ── 4. Enrich with audio features ─────────────────────────────────────────
    _enrich_with_audio_features(sp, tracks)

    logger.info(
        "Spotify: fetched %d tracks from category '%s' (playlist: '%s')",
        len(tracks),
        category_id,
        playlist.get("name", ""),
    )
    return tracks


def get_featured_tracks(limit: int = 20, country: str = "US") -> list[SpotifyTrack]:
    """
    Pull tracks from Spotify's global featured playlists.
    Good fallback when SPOTIFY_CATEGORY_POOL is empty or an API call fails.
    """
    sp = _get_client()
    try:
        featured = sp.featured_playlists(country=country, limit=5)
        playlists = (featured.get("playlists") or {}).get("items") or []
        if not playlists:
            return []
        playlist = random.choice(playlists)
        result = sp.playlist_tracks(
            playlist["id"],
            fields=(
                "items(track(id,name,artists,album,external_urls,"
                "preview_url,popularity,duration_ms))"
            ),
            limit=min(limit * 2, 50),
        )
        raw_items = (result or {}).get("items") or []
    except Exception as exc:
        logger.error("Spotify featured_playlists failed: %s", exc)
        return []

    tracks: list[SpotifyTrack] = []
    for item in raw_items:
        t = _parse_track_item(item)
        if t and t.duration_ms > 0:
            t.genres = ["featured"]
            tracks.append(t)
        if len(tracks) >= limit:
            break

    if not tracks:
        return []

    _enrich_with_audio_features(sp, tracks)
    logger.info("Spotify: fetched %d featured tracks", len(tracks))
    return tracks
