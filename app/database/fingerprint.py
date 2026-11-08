"""
Acoustic Landmark Constellation Fingerprinting (Shazam-style algorithm).

Extracts prominent spectral energy peaks, forms (f1, f2, dt) landmark pairs,
and hashes them into an inverted index. Enables sub-100ms song identification
and exact temporal alignment under mobile microphone noise and room reverberation.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

from app.config import settings

logger = logging.getLogger(__name__)

DB_PATH = Path(getattr(settings, "FINGERPRINT_DB_PATH", "./fingerprints.db"))


class LandmarkFingerprinter:
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        db_path: Path | str | None = None,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.db_path = str(db_path or DB_PATH)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS songs (
                    song_id TEXT PRIMARY KEY,
                    title TEXT,
                    artist TEXT,
                    album TEXT,
                    store_url TEXT,
                    cover_url TEXT,
                    duration_s REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    hash INTEGER,
                    song_id TEXT,
                    offset_ds INTEGER,
                    FOREIGN KEY(song_id) REFERENCES songs(song_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)"
            )

    def _extract_peaks(
        self, y: np.ndarray, percentile: float = 80.0, filter_size: int = 15
    ) -> list[tuple[int, int]]:
        """Compute spectrogram and find local 2D peak landmarks."""
        freqs, times, Sxx = signal.spectrogram(
            y,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        Sxx_log = 10 * np.log10(np.maximum(Sxx, 1e-10))

        # Dynamic threshold based on active signal energy
        active_vals = Sxx_log[Sxx_log > -90]
        amp_thresh = (
            float(np.percentile(active_vals, percentile))
            if len(active_vals) > 0
            else -50.0
        )

        # Local 2D maximum filter
        local_max = maximum_filter(Sxx_log, size=filter_size) == Sxx_log
        background = Sxx_log >= amp_thresh
        eroded_background = local_max & background

        f_idx, t_idx = np.where(eroded_background)
        peaks = list(zip(f_idx, t_idx))
        peaks.sort(key=lambda p: p[1])
        return peaks

    def fingerprint(self, y: np.ndarray) -> list[tuple[int, float]]:
        """
        Generate pair hashes: (f1, f2, delta_time) -> 32-bit hash.
        Returns list of (hash, time_seconds).
        """
        peaks = self._extract_peaks(y)
        hashes = []
        fan_value = 15  # Target pairs per anchor peak
        max_time_delta = 100  # ~2.3 seconds
        min_time_delta = 5  # ~0.1 seconds

        for i in range(len(peaks)):
            f1, t1 = peaks[i]
            for j in range(1, fan_value):
                if i + j < len(peaks):
                    f2, t2 = peaks[i + j]
                    dt = t2 - t1
                    if min_time_delta <= dt <= max_time_delta:
                        # 32-bit integer hash: pack f1 (10 bits), f2 (10 bits), dt (12 bits)
                        h = (
                            (int(f1) & 0x3FF) << 22
                            | (int(f2) & 0x3FF) << 12
                            | (int(dt) & 0xFFF)
                        )
                        time_sec = t1 * self.hop_length / self.sr
                        hashes.append((h, time_sec))
        return hashes

    def index_song(
        self,
        song_id: str,
        y: np.ndarray,
        meta: dict[str, Any],
    ) -> int:
        """Indexes a song's audio landmarks into the database."""
        t0 = time.perf_counter()
        hashes = self.fingerprint(y)
        duration_s = len(y) / self.sr

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO songs (song_id, title, artist, album, store_url, cover_url, duration_s)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    song_id,
                    meta.get("title", "Unknown"),
                    meta.get("artist", "Unknown"),
                    meta.get("album", ""),
                    meta.get("store_url", ""),
                    meta.get("cover_url", ""),
                    duration_s,
                ),
            )

            # Store offset in deciseconds (0.1s units) for efficient integer indexing
            rows = [
                (h, song_id, int(round(t_sec * 10))) for h, t_sec in hashes
            ]
            conn.executemany(
                "INSERT INTO fingerprints (hash, song_id, offset_ds) VALUES (?, ?, ?)",
                rows,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Indexed '%s - %s' (%d landmark hashes in %.1fms)",
            meta.get("artist"),
            meta.get("title"),
            len(hashes),
            elapsed_ms,
        )
        return len(hashes)

    def match_snippet(self, snippet: np.ndarray) -> dict[str, Any]:
        """
        Query an audio snippet against the constellation index.
        Finds the song and the exact temporal alignment with highest vote count.
        """
        t0 = time.perf_counter()
        snippet_hashes = self.fingerprint(snippet)

        if not snippet_hashes:
            return {"match": None, "query_time_ms": 0.0}

        hash_list = [h for h, _ in snippet_hashes]
        # Query matching hashes in chunks to stay within SQLite variable limits
        chunk_size = 900
        matches: dict[str, dict[int, int]] = {}

        with self._get_conn() as conn:
            for i in range(0, len(hash_list), chunk_size):
                chunk = hash_list[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT hash, song_id, offset_ds FROM fingerprints WHERE hash IN ({placeholders})",
                    chunk,
                )

                # Map snippet time to matched database rows
                # Pre-build lookup for snippet hashes
                time_lookup: dict[int, list[float]] = {}
                for h, t in snippet_hashes[i : i + chunk_size]:
                    time_lookup.setdefault(h, []).append(t)

                for h, song_id, song_offset_ds in cursor.fetchall():
                    for t_snippet in time_lookup.get(h, []):
                        snippet_offset_ds = int(round(t_snippet * 10))
                        delta_ds = song_offset_ds - snippet_offset_ds
                        if song_id not in matches:
                            matches[song_id] = {}
                        matches[song_id][delta_ds] = (
                            matches[song_id].get(delta_ds, 0) + 1
                        )

        query_time_ms = (time.perf_counter() - t0) * 1000

        if not matches:
            return {"match": None, "query_time_ms": round(query_time_ms, 2)}

        # Find song and offset with maximum co-occurring aligned matches
        best_song = None
        best_offset_ds = 0
        max_votes = 0

        for song_id, offsets in matches.items():
            for offset_ds, votes in offsets.items():
                if votes > max_votes:
                    max_votes = votes
                    best_song = song_id
                    best_offset_ds = offset_ds

        # Minimum confidence threshold (at least 8 co-occurring aligned peaks)
        if max_votes < 8 or not best_song:
            return {
                "match": None,
                "confidence_votes": max_votes,
                "query_time_ms": round(query_time_ms, 2),
            }

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT title, artist, album, store_url, cover_url, duration_s FROM songs WHERE song_id = ?",
                (best_song,),
            ).fetchone()

        meta = {}
        if row:
            meta = {
                "title": row[0],
                "artist": row[1],
                "album": row[2],
                "store_url": row[3],
                "cover_url": row[4],
                "duration_s": row[5],
            }

        return {
            "match": {
                "song_id": best_song,
                "title": meta.get("title"),
                "artist": meta.get("artist"),
                "album": meta.get("album"),
                "links": {"store": meta.get("store_url")},
                "cover_url": meta.get("cover_url"),
                "heard_at_second": round(best_offset_ds / 10.0, 2),
                "confidence_votes": max_votes,
            },
            "total_snippet_hashes": len(snippet_hashes),
            "query_time_ms": round(query_time_ms, 2),
        }


# Singleton instance
_fingerprinter: LandmarkFingerprinter | None = None


def get_fingerprinter() -> LandmarkFingerprinter:
    global _fingerprinter
    if _fingerprinter is None:
        _fingerprinter = LandmarkFingerprinter()
    return _fingerprinter
