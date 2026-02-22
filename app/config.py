from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Qdrant ────────────────────────────────────────────────────────────────
    # Cloud mode  : set QDRANT_URL (e.g. https://xyz.qdrant.io) + QDRANT_API_KEY
    # Local mode  : set QDRANT_HOST + QDRANT_PORT (default: localhost:6333)
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "orpheus_songs")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "8000"))

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # cron-job.org hits /scheduler/trigger every 15 min from outside.
    # The in-process scheduler fires every SCHEDULE_MINUTES as a backup.
    SONGS_PER_RUN: int = int(os.getenv("SONGS_PER_RUN", "4"))
    SCHEDULE_MINUTES: int = int(os.getenv("SCHEDULE_MINUTES", "30"))

    # Scratch directory for temporary audio files (deleted after each run)
    SCRATCH_DIR: Path = Path(os.getenv("SCRATCH_DIR", "./scratch"))

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Song-level: 7 VibeNet + 8 librosa scalars + 2 spectral (flux+harmonic) + 13 MFCCs
    #   = 30 total  (must stay in sync with extractor.py SongFeatures)
    EMBEDDING_DIM: int = 30

    # Frame-level (Shazam-style snippet matching):
    #   13 MFCCs + 12 chroma + 5 spectral scalars + 1 flux + 1 harmonic = 32
    FRAME_COLLECTION: str = os.getenv("FRAME_COLLECTION", "orpheus_frames")
    FRAME_EMBEDDING_DIM: int = 32
    FRAME_WINDOW_S: float = 5.0  # seconds per analysis window
    FRAME_HOP_S: float = 2.5  # hop between window starts (50 % overlap)

    # ── Spotify (primary song source) ─────────────────────────────────────────
    # Get credentials at https://developer.spotify.com/dashboard
    # If not set the pipeline falls back to the YouTube query pool below.
    # ── HF Extraction Space ────────────────────────────────────────────────
    # Audio feature extraction is offloaded to a Hugging Face Space to avoid
    # OOM on Heroku's 512 MB dynos.  Set the URL to your Space's root.
    HF_EXTRACTION_URL: str = os.getenv(
        "HF_EXTRACTION_URL", "https://mryamusa-orpheus-extractor.hf.space"
    )
    HF_EXTRACTION_SECRET: str | None = os.getenv("HF_EXTRACTION_SECRET")
    # Hugging Face API token — required for private Spaces
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")

    SPOTIFY_CLIENT_ID: str | None = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET: str | None = os.getenv("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_COUNTRY: str = os.getenv("SPOTIFY_COUNTRY", "US")

    # Spotify category IDs to rotate through each run.
    # See app/ingestion/spotify_client.py → SPOTIFY_CATEGORIES for the full list.
    SPOTIFY_CATEGORY_POOL: list[str] = [
        "hiphop",
        "pop",
        "rnb",
        "latin",
        "edm_dance",
        "rock",
        "afro",
        "indie_alt",
        "soul",
        "workout",
        "party",
        "chill",
        "romance",
        "mood",
    ]

    # ── YouTube fallback query pool ────────────────────────────────────────────
    # Used when Spotify credentials are not configured.
    YOUTUBE_QUERY_POOL: list[str] = [
        "trap hip hop 2025 popular",
        "best new drill rap 2025",
        "melodic rap hits 2025",
        "pop hits 2025 official audio",
        "dark pop aesthetic 2025",
        "deep house music mix 2025",
        "future bass electronic 2025",
        "lo fi hip hop chill beats",
        "phonk drift music 2025",
        "afrobeats hit song 2025",
        "alternative rock indie 2025",
        "rnb love song 2025",
        "neo soul smooth 2025",
        "reggaeton hit 2025",
        "sad emotional music 2025",
        "hype workout music 2025",
        "late night vibe song 2025",
        "happy feel good song 2025",
    ]


settings = Settings()
