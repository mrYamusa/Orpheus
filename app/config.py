from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "orpheus_songs")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

    # Ingestion scheduler
    SONGS_PER_RUN: int = int(os.getenv("SONGS_PER_RUN", "2"))
    SCHEDULE_HOURS: int = int(os.getenv("SCHEDULE_HOURS", "3"))

    # Scratch directory for temporary audio files
    SCRATCH_DIR: Path = Path(os.getenv("SCRATCH_DIR", "./scratch"))

    # Song embedding vector dimensions (must stay in sync with extractor.py)
    EMBEDDING_DIM: int = 28  # 7 vibenet + 8 librosa scalars + 13 mfcc means

    # Frame-level (snippet matching) collection
    FRAME_COLLECTION: str = os.getenv("FRAME_COLLECTION", "orpheus_frames")
    # 13 MFCCs + 12 chroma bins + 5 scalars (rms, zcr, centroid, bandwidth, rolloff)
    FRAME_EMBEDDING_DIM: int = 30
    FRAME_WINDOW_S: float = 5.0  # seconds per analysis window
    FRAME_HOP_S: float = 2.5  # hop between window starts (50% overlap)

    # Pool of YouTube search queries used by the background ingestion job.
    # Queries are randomly sampled from this list each run so the library
    # grows across many genres and moods over time.
    YOUTUBE_QUERY_POOL: list[str] = [
        # Hip-hop / Trap
        "trap hip hop 2024 popular",
        "best new drill rap 2024",
        "underground hip hop chill 2024",
        "melodic rap hits 2025",
        # Pop
        "pop hits 2025 official audio",
        "indie pop song 2024",
        "dark pop aesthetic 2024",
        # Electronic / Dance
        "deep house music mix",
        "future bass electronic 2024",
        "lo fi hip hop chill beats",
        "phonk drift music 2024",
        "afrobeats hit song 2024",
        # Alternative / Rock
        "alternative rock indie 2024",
        "bedroom pop dreamy 2024",
        # R&B / Soul
        "rnb love song 2024",
        "neo soul smooth 2024",
        "afro soul hit 2024",
        # Latin / World
        "reggaeton hit 2024",
        "latin pop 2025 oficial",
        # Mood-based
        "sad emotional music 2024",
        "hype workout music 2024",
        "late night vibe song 2024",
        "happy feel good song 2024",
    ]


settings = Settings()
