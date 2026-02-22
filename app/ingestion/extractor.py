"""
Audio feature extractor — HTTP proxy to Hugging Face Space.

All heavy ML work (VibeNet, librosa, ONNX) runs on a Hugging Face Space
with 16 GB RAM.  This module uploads the audio file, receives JSON, and
reconstructs the dataclasses that the rest of the pipeline expects.

Public API (unchanged from the local version)
──────────
  extract_all(path)               → (SongFeatures, list[FrameFeatures])
  extract_snippet_embedding(path) → list[float]  (32-dim)
  unload_model()                  → no-op (model lives on HF Space)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Normalisation bounds (only needed for SongFeatures.to_embedding) ─────────
_TEMPO_MIN, _TEMPO_MAX = 40.0, 220.0
_RMS_MIN, _RMS_MAX = 0.0, 0.5
_CENTROID_MIN, _CENTROID_MAX = 200.0, 8000.0
_BANDWIDTH_MIN, _BANDWIDTH_MAX = 200.0, 5000.0
_ROLLOFF_MIN, _ROLLOFF_MAX = 500.0, 11025.0
_ZCR_MIN, _ZCR_MAX = 0.0, 0.3
_FLUX_MIN, _FLUX_MAX = 0.0, 10.0

_MFCC_SCALES = np.array(
    [150, 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], dtype=np.float32,
)
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_HTTP_TIMEOUT = 300.0  # 5 min — extraction can be slow on free CPU


def _clamp_norm(val: float, lo: float, hi: float) -> float:
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _norm_mfcc(arr: np.ndarray) -> np.ndarray:
    return np.tanh(arr / _MFCC_SCALES) * 0.5 + 0.5


def unload_model() -> None:
    """No-op — model lives on the HF Space, not here."""
    pass


# ─────────────────────────── Data classes ─────────────────────────────────────


@dataclass
class SongFeatures:
    """Whole-song aggregate features → 30-dim embedding."""

    acousticness: float = 0.0
    danceability: float = 0.0
    energy: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    valence: float = 0.0

    tempo_bpm: float = 0.0
    key: str = "C"
    mode: str = "major"
    rms_mean: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zcr_mean: float = 0.0

    spectral_flux_mean: float = 0.0
    spectral_flux_std: float = 0.0
    harmonic_ratio: float = 0.0

    mfcc_means: list[float] = field(default_factory=lambda: [0.0] * 13)
    instrument_profile: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_embedding(self) -> list[float]:
        key_idx = _KEY_NAMES.index(self.key) if self.key in _KEY_NAMES else 0
        return [
            self.acousticness,
            self.danceability,
            self.energy,
            self.instrumentalness,
            self.liveness,
            self.speechiness,
            self.valence,
            _clamp_norm(self.tempo_bpm, _TEMPO_MIN, _TEMPO_MAX),
            key_idx / 11.0,
            1.0 if self.mode == "major" else 0.0,
            _clamp_norm(self.rms_mean, _RMS_MIN, _RMS_MAX),
            _clamp_norm(self.spectral_centroid_mean, _CENTROID_MIN, _CENTROID_MAX),
            _clamp_norm(self.spectral_bandwidth_mean, _BANDWIDTH_MIN, _BANDWIDTH_MAX),
            _clamp_norm(self.spectral_rolloff_mean, _ROLLOFF_MIN, _ROLLOFF_MAX),
            _clamp_norm(self.zcr_mean, _ZCR_MIN, _ZCR_MAX),
            _clamp_norm(self.spectral_flux_mean, _FLUX_MIN, _FLUX_MAX),
            float(np.clip(self.harmonic_ratio, 0.0, 1.0)),
            *_norm_mfcc(np.array(self.mfcc_means, dtype=np.float32)).tolist(),
        ]


@dataclass
class FrameFeatures:
    """Per-window features → 32-dim embedding (pre-computed by HF Space)."""

    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    mfcc_means: list[float] = field(default_factory=lambda: [0.0] * 13)
    chroma_means: list[float] = field(default_factory=lambda: [0.0] * 12)

    rms_mean: float = 0.0
    zcr_mean: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_flux_mean: float = 0.0
    harmonic_ratio: float = 0.0

    _embedding: list[float] = field(default_factory=list)

    def to_embedding(self) -> list[float]:
        if self._embedding:
            return self._embedding
        # Fallback — compute locally (should not happen in normal flow)
        return [
            *_norm_mfcc(np.array(self.mfcc_means, dtype=np.float32)).tolist(),
            *[float(v) for v in self.chroma_means],
            _clamp_norm(self.rms_mean, _RMS_MIN, _RMS_MAX),
            _clamp_norm(self.zcr_mean, _ZCR_MIN, _ZCR_MAX),
            _clamp_norm(self.spectral_centroid_mean, _CENTROID_MIN, _CENTROID_MAX),
            _clamp_norm(self.spectral_bandwidth_mean, _BANDWIDTH_MIN, _BANDWIDTH_MAX),
            _clamp_norm(self.spectral_rolloff_mean, _ROLLOFF_MIN, _ROLLOFF_MAX),
            _clamp_norm(self.spectral_flux_mean, _FLUX_MIN, _FLUX_MAX),
            float(np.clip(self.harmonic_ratio, 0.0, 1.0)),
        ]


# ─────────────────────────── HTTP helpers ─────────────────────────────────────


def _hf_url(endpoint: str) -> str:
    base = settings.HF_EXTRACTION_URL.rstrip("/")
    return f"{base}/{endpoint.lstrip('/')}"


def _hf_headers() -> dict:
    """Auth headers for private HF Spaces (Bearer token)."""
    headers: dict[str, str] = {}
    if settings.HF_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
    return headers


def _hf_params() -> dict:
    """Query params to include in every request (app-level secret)."""
    if settings.HF_EXTRACTION_SECRET:
        return {"secret": settings.HF_EXTRACTION_SECRET}
    return {}


# ─────────────────────────── Public API ───────────────────────────────────────


async def extract_all(
    audio_path: str | Path,
) -> tuple[SongFeatures, list[FrameFeatures]]:
    """
    Upload audio to HF Space /extract endpoint, parse response into
    (SongFeatures, list[FrameFeatures]).
    """
    audio_path = Path(audio_path)
    logger.info("Sending to HF extractor: %s", audio_path.name)

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=_hf_headers()) as client:
        with open(audio_path, "rb") as f:
            resp = await client.post(
                _hf_url("/extract"),
                files={"file": (audio_path.name, f, "audio/mpeg")},
                params=_hf_params(),
            )

    if resp.status_code != 200:
        raise RuntimeError(
            f"HF extractor returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    song_data = data["song"]
    frames_data = data["frames"]

    song = SongFeatures(
        acousticness=song_data["acousticness"],
        danceability=song_data["danceability"],
        energy=song_data["energy"],
        instrumentalness=song_data["instrumentalness"],
        liveness=song_data["liveness"],
        speechiness=song_data["speechiness"],
        valence=song_data["valence"],
        tempo_bpm=song_data["tempo_bpm"],
        key=song_data["key"],
        mode=song_data["mode"],
        rms_mean=song_data["rms_mean"],
        spectral_centroid_mean=song_data["spectral_centroid_mean"],
        spectral_bandwidth_mean=song_data["spectral_bandwidth_mean"],
        spectral_rolloff_mean=song_data["spectral_rolloff_mean"],
        zcr_mean=song_data["zcr_mean"],
        spectral_flux_mean=song_data["spectral_flux_mean"],
        spectral_flux_std=song_data["spectral_flux_std"],
        harmonic_ratio=song_data["harmonic_ratio"],
        mfcc_means=song_data["mfcc_means"],
        instrument_profile=song_data["instrument_profile"],
        duration_s=song_data["duration_s"],
    )

    frames: list[FrameFeatures] = []
    for fd in frames_data:
        frames.append(FrameFeatures(
            timestamp_start=fd["timestamp_start"],
            timestamp_end=fd["timestamp_end"],
            mfcc_means=fd["mfcc_means"],
            chroma_means=fd["chroma_means"],
            rms_mean=fd["rms_mean"],
            zcr_mean=fd["zcr_mean"],
            spectral_centroid_mean=fd["spectral_centroid_mean"],
            spectral_bandwidth_mean=fd["spectral_bandwidth_mean"],
            spectral_rolloff_mean=fd["spectral_rolloff_mean"],
            spectral_flux_mean=fd["spectral_flux_mean"],
            harmonic_ratio=fd["harmonic_ratio"],
            _embedding=fd.get("embedding", []),
        ))

    logger.info(
        "  → HF returned %d-dim song, %d frames (%.1fs)",
        len(song.to_embedding()),
        len(frames),
        song.duration_s,
    )
    return song, frames


async def extract_snippet_embedding(audio_path: str | Path) -> list[float]:
    """
    Upload a short audio clip to HF Space /extract-snippet, get 32-dim embedding.
    """
    audio_path = Path(audio_path)
    logger.info("Sending snippet to HF extractor: %s", audio_path.name)

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=_hf_headers()) as client:
        with open(audio_path, "rb") as f:
            resp = await client.post(
                _hf_url("/extract-snippet"),
                files={"file": (audio_path.name, f, "audio/mpeg")},
                params=_hf_params(),
            )

    if resp.status_code != 200:
        raise RuntimeError(
            f"HF extractor returned {resp.status_code}: {resp.text[:500]}"
        )

    return resp.json()["embedding"]
