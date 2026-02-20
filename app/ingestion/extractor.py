"""
Audio feature extractor.

Produces two types of output from a single audio load:

  SongFeatures (28-dim embedding)
  ─────────────────────────────────
  Global (whole-song) averages used for mood/vibe search.
    - 7  VibeNet (EfficientNet ONNX): acousticness, danceability, energy,
         instrumentalness, liveness, speechiness, valence
    - 8  librosa scalars: tempo, key, mode, rms, centroid, bandwidth, rolloff, zcr
    - 13 MFCC coefficient means

  FrameFeatures (30-dim embedding per window)
  ─────────────────────────────────────────────
  5-second windows (2.5s hop) over the full song used for snippet matching.
  Each window stores its timestamps so a short audio clip can be located
  inside a song (Shazam-style).
    - 13 MFCC means
    - 12 chroma bin means
    -  5 scalar means: rms, zcr, centroid, bandwidth, rolloff

Public API
──────────
  extract_all(path)              → (SongFeatures, list[FrameFeatures])
  extract_snippet_embedding(path) → list[float]   (30-dim, for snippet search)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from vibenet import load_model
from vibenet.core import InferenceResult

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Normalisation bounds for librosa features ────────────────────────────────
_TEMPO_MIN, _TEMPO_MAX = 40.0, 220.0
_RMS_MIN, _RMS_MAX = 0.0, 0.5
_CENTROID_MIN, _CENTROID_MAX = 200.0, 8000.0
_BANDWIDTH_MIN, _BANDWIDTH_MAX = 200.0, 5000.0
_ROLLOFF_MIN, _ROLLOFF_MAX = 500.0, 11025.0
_ZCR_MIN, _ZCR_MAX = 0.0, 0.3

# MFCC per-coefficient tanh scale so output lands in (0, 1)
_MFCC_SCALES = np.array(
    [150.0, 60.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
    dtype=np.float32,
)

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# librosa hop length used for all feature matrices (≈23ms per frame at 22050 Hz)
_HOP_LENGTH = 512
_SR = 22050

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _clamp_norm(val: float, lo: float, hi: float) -> float:
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _norm_mfcc(arr: np.ndarray) -> np.ndarray:
    """tanh-squash an (13,) array → (0, 1)."""
    return np.tanh(arr / _MFCC_SCALES) * 0.5 + 0.5


# ─────────────────────────── Data classes ─────────────────────────────────────


@dataclass
class SongFeatures:
    """Whole-song aggregate features + 28-dim embedding."""

    # VibeNet
    acousticness: float = 0.0
    danceability: float = 0.0
    energy: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    speechiness: float = 0.0
    valence: float = 0.0

    # Librosa scalars
    tempo_bpm: float = 0.0
    key: str = "C"
    mode: str = "major"
    rms_mean: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zcr_mean: float = 0.0

    # MFCCs
    mfcc_means: list[float] = field(default_factory=lambda: [0.0] * 13)

    # Metadata
    duration_s: float = 0.0

    def to_embedding(self) -> list[float]:
        """
        28-dim normalised vector:
          [0-6]   VibeNet (already 0-1)
          [7]     tempo
          [8]     key index / 11
          [9]     mode (1=major, 0=minor)
          [10]    rms
          [11]    centroid
          [12]    bandwidth
          [13]    rolloff
          [14]    zcr
          [15-27] 13 MFCCs
        """
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
            *_norm_mfcc(np.array(self.mfcc_means, dtype=np.float32)).tolist(),
        ]

    def to_payload_dict(self) -> dict[str, Any]:
        return {
            "acousticness": self.acousticness,
            "danceability": self.danceability,
            "energy": self.energy,
            "instrumentalness": self.instrumentalness,
            "liveness": self.liveness,
            "speechiness": self.speechiness,
            "valence": self.valence,
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "mode": self.mode,
            "rms_mean": self.rms_mean,
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "spectral_bandwidth_mean": self.spectral_bandwidth_mean,
            "spectral_rolloff_mean": self.spectral_rolloff_mean,
            "zcr_mean": self.zcr_mean,
            "mfcc_means": self.mfcc_means,
            "duration_s": self.duration_s,
        }


@dataclass
class FrameFeatures:
    """
    Features for one time window of a song.

    30-dim embedding:
      [0-12]  13 MFCC means  (tanh-squashed → [0,1])
      [13-24] 12 chroma means (librosa output already [0,1])
      [25]    rms_mean
      [26]    zcr_mean
      [27]    spectral_centroid_mean
      [28]    spectral_bandwidth_mean
      [29]    spectral_rolloff_mean
    """

    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    mfcc_means: list[float] = field(default_factory=lambda: [0.0] * 13)
    chroma_means: list[float] = field(default_factory=lambda: [0.0] * 12)
    rms_mean: float = 0.0
    zcr_mean: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0

    def to_embedding(self) -> list[float]:
        return [
            *_norm_mfcc(np.array(self.mfcc_means, dtype=np.float32)).tolist(),
            *[float(v) for v in self.chroma_means],  # chroma is already [0,1]
            _clamp_norm(self.rms_mean, _RMS_MIN, _RMS_MAX),
            _clamp_norm(self.zcr_mean, _ZCR_MIN, _ZCR_MAX),
            _clamp_norm(self.spectral_centroid_mean, _CENTROID_MIN, _CENTROID_MAX),
            _clamp_norm(self.spectral_bandwidth_mean, _BANDWIDTH_MIN, _BANDWIDTH_MAX),
            _clamp_norm(self.spectral_rolloff_mean, _ROLLOFF_MIN, _ROLLOFF_MAX),
        ]


# ─────────────────────── Internal helpers ─────────────────────────────────────


def _detect_key_mode(chroma: np.ndarray) -> tuple[str, str]:
    """Return (key_name, 'major'|'minor') using Krumhansl-Kessler profiles."""
    avg = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(avg))
    rot_maj = np.roll(_MAJOR_PROFILE, -key_idx)
    rot_min = np.roll(_MINOR_PROFILE, -key_idx)
    mode = "major" if np.dot(avg, rot_maj) >= np.dot(avg, rot_min) else "minor"
    return _KEY_NAMES[key_idx], mode


def _build_frames(
    mfcc: np.ndarray,  # (13, T)
    chroma: np.ndarray,  # (12, T)
    rms: np.ndarray,  # (1, T)
    zcr: np.ndarray,  # (1, T)
    centroid: np.ndarray,  # (1, T)
    bandwidth: np.ndarray,  # (1, T)
    rolloff: np.ndarray,  # (1, T)
    times: np.ndarray,  # (T,)
) -> list[FrameFeatures]:
    """
    Slice all feature matrices into overlapping windows and return
    a FrameFeatures for each window.
    """
    n_frames = times.shape[0]
    frames_per_win = max(1, int(settings.FRAME_WINDOW_S * _SR / _HOP_LENGTH))
    hop_frames = max(1, int(settings.FRAME_HOP_S * _SR / _HOP_LENGTH))

    result: list[FrameFeatures] = []
    start = 0
    while start < n_frames:
        end = min(start + frames_per_win, n_frames)
        sl = slice(start, end)

        t_start = float(times[start])
        t_end = float(times[end - 1])

        result.append(
            FrameFeatures(
                timestamp_start=t_start,
                timestamp_end=t_end,
                mfcc_means=np.mean(mfcc[:, sl], axis=1).tolist(),
                chroma_means=np.mean(chroma[:, sl], axis=1).tolist(),
                rms_mean=float(np.mean(rms[0, sl])),
                zcr_mean=float(np.mean(zcr[0, sl])),
                spectral_centroid_mean=float(np.mean(centroid[0, sl])),
                spectral_bandwidth_mean=float(np.mean(bandwidth[0, sl])),
                spectral_rolloff_mean=float(np.mean(rolloff[0, sl])),
            )
        )

        if end >= n_frames:
            break
        start += hop_frames

    return result


def _compute_matrices_sync(y: np.ndarray, sr: int) -> dict:
    """Compute all librosa feature matrices in one synchronous call (runs in thread)."""
    return {
        "tempo": librosa.beat.beat_track(y=y, sr=sr, hop_length=_HOP_LENGTH),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=_HOP_LENGTH),
        "chroma": librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP_LENGTH),
        "rms": librosa.feature.rms(y=y, hop_length=_HOP_LENGTH),
        "centroid": librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=_HOP_LENGTH
        ),
        "bandwidth": librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=_HOP_LENGTH
        ),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=_HOP_LENGTH),
        "zcr": librosa.feature.zero_crossing_rate(y, hop_length=_HOP_LENGTH),
    }


# ─────────────────────────── Public API ───────────────────────────────────────


async def extract_all(
    audio_path: str | Path,
) -> tuple[SongFeatures, list[FrameFeatures]]:
    """
    Single-pass extraction.

    Loads audio once, computes all librosa matrices once (in a thread),
    then derives both:
      - SongFeatures  (whole-song averages)
      - list[FrameFeatures]  (one entry per 5-second window)

    VibeNet runs in parallel with all librosa work.
    """
    audio_path = str(audio_path)
    model = _get_model()
    logger.info("Extracting features: %s", audio_path)

    # Stage 1 — VibeNet + audio load (concurrent)
    async with asyncio.TaskGroup() as tg:
        t_vibe = tg.create_task(asyncio.to_thread(model.predict, audio_path))
        t_load = tg.create_task(asyncio.to_thread(librosa.load, audio_path, sr=_SR))

    vibe: InferenceResult = t_vibe.result()[0]
    y, sr = t_load.result()

    # Stage 2 — all librosa matrices in a single thread call
    mats = await asyncio.to_thread(_compute_matrices_sync, y, sr)

    mfcc = mats["mfcc"]  # (13, T)
    chroma = mats["chroma"]  # (12, T)
    rms = mats["rms"]  # (1, T)
    centroid = mats["centroid"]  # (1, T)
    bandwidth = mats["bandwidth"]  # (1, T)
    rolloff = mats["rolloff"]  # (1, T)
    zcr = mats["zcr"]  # (1, T)
    tempo_val, _ = mats["tempo"]

    times = librosa.times_like(rms, sr=sr, hop_length=_HOP_LENGTH)  # (T,)

    # ── Song-level aggregates ──────────────────────────────────────────────────
    tempo_bpm = float(np.atleast_1d(tempo_val)[0])
    key_name, mode = _detect_key_mode(chroma)
    duration_s = float(librosa.get_duration(y=y, sr=sr))

    song = SongFeatures(
        acousticness=vibe.acousticness,
        danceability=vibe.danceability,
        energy=vibe.energy,
        instrumentalness=vibe.instrumentalness,
        liveness=vibe.liveness,
        speechiness=vibe.speechiness,
        valence=vibe.valence,
        tempo_bpm=tempo_bpm,
        key=key_name,
        mode=mode,
        rms_mean=float(np.mean(rms)),
        spectral_centroid_mean=float(np.mean(centroid)),
        spectral_bandwidth_mean=float(np.mean(bandwidth)),
        spectral_rolloff_mean=float(np.mean(rolloff)),
        zcr_mean=float(np.mean(zcr)),
        mfcc_means=np.mean(mfcc, axis=1).tolist(),
        duration_s=duration_s,
    )

    # ── Frame-level windows ────────────────────────────────────────────────────
    frames = _build_frames(mfcc, chroma, rms, zcr, centroid, bandwidth, rolloff, times)
    logger.info(
        "  → %d song-level dims, %d frames", len(song.to_embedding()), len(frames)
    )

    return song, frames


async def extract_snippet_embedding(audio_path: str | Path) -> list[float]:
    """
    Extract a 30-dim frame embedding from a short audio clip.

    Used by POST /match/snippet: load the clip, compute feature matrices,
    average across ALL frames (entire clip = one 'super-frame'), return
    the normalised 30-dim vector ready for Qdrant frame search.
    """
    audio_path = str(audio_path)
    y, sr = await asyncio.to_thread(librosa.load, audio_path, sr=_SR)
    mats = await asyncio.to_thread(_compute_matrices_sync, y, sr)

    snippet = FrameFeatures(
        timestamp_start=0.0,
        timestamp_end=float(librosa.get_duration(y=y, sr=sr)),
        mfcc_means=np.mean(mats["mfcc"], axis=1).tolist(),
        chroma_means=np.mean(mats["chroma"], axis=1).tolist(),
        rms_mean=float(np.mean(mats["rms"])),
        zcr_mean=float(np.mean(mats["zcr"])),
        spectral_centroid_mean=float(np.mean(mats["centroid"])),
        spectral_bandwidth_mean=float(np.mean(mats["bandwidth"])),
        spectral_rolloff_mean=float(np.mean(mats["rolloff"])),
    )
    return snippet.to_embedding()
