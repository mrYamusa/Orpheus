"""
Orpheus feature-extraction micro-service (Hugging Face Space).

Endpoints
─────────
  POST /extract          – upload mp3 → full SongFeatures + FrameFeatures JSON
  POST /extract-snippet  – upload mp3 → 32-dim snippet embedding JSON
  GET  /health           – liveness check
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import uuid
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from vibenet import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Orpheus Extractor", version="1.0.0")

# ── Auth ──────────────────────────────────────────────────────────────────────
_API_SECRET = os.getenv("API_SECRET", "")

# ── Constants ─────────────────────────────────────────────────────────────────
_SR = 22050
_HOP_LENGTH = 512
_MAX_DURATION_S = 180.0  # generous here — HF has plenty of RAM

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
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Frame windowing defaults (must match Heroku config)
FRAME_WINDOW_S = float(os.getenv("FRAME_WINDOW_S", "5.0"))
FRAME_HOP_S = float(os.getenv("FRAME_HOP_S", "2.5"))

# ── Singleton model ──────────────────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading VibeNet model …")
        _model = load_model()
    return _model


# ── Helper functions ──────────────────────────────────────────────────────────

def _clamp_norm(val: float, lo: float, hi: float) -> float:
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _detect_key_mode(chroma: np.ndarray) -> tuple[str, str]:
    avg = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(avg))
    rot_maj = np.roll(_MAJOR_PROFILE, -key_idx)
    rot_min = np.roll(_MINOR_PROFILE, -key_idx)
    mode = "major" if np.dot(avg, rot_maj) >= np.dot(avg, rot_min) else "minor"
    return _KEY_NAMES[key_idx], mode


def _estimate_instrument_profile(
    y: np.ndarray, harmonic: np.ndarray, percussive: np.ndarray,
    chroma: np.ndarray, centroid: np.ndarray,
) -> dict:
    total_rms = float(np.sqrt(np.mean(y**2))) + 1e-8
    harm_rms = float(np.sqrt(np.mean(harmonic**2)))
    perc_rms = float(np.sqrt(np.mean(percussive**2)))

    harmonic_ratio = float(np.clip(harm_rms / total_rms, 0.0, 1.0))
    percussive_ratio = float(np.clip(perc_rms / total_rms, 0.0, 1.0))
    tonal_strength = float(np.mean(np.max(chroma, axis=0)))
    brightness = float(np.clip(np.mean(centroid) / (_SR / 2), 0.0, 1.0))

    if harmonic_ratio > 0.80 and tonal_strength > 0.70:
        label = "orchestral / strings"
    elif harmonic_ratio > 0.70 and brightness < 0.25:
        label = "piano / keyboard"
    elif harmonic_ratio > 0.65 and 0.25 <= brightness < 0.45:
        label = "guitar / plucked"
    elif percussive_ratio > 0.55:
        label = "drums / percussion"
    elif harmonic_ratio > 0.55 and brightness > 0.45:
        label = "electronic / synth lead"
    elif harmonic_ratio < 0.40 and brightness < 0.30:
        label = "synthetic / pad"
    else:
        label = "mixed"

    return {
        "harmonic_ratio": round(harmonic_ratio, 4),
        "percussive_ratio": round(percussive_ratio, 4),
        "tonal_strength": round(tonal_strength, 4),
        "brightness": round(brightness, 4),
        "estimated_profile": label,
    }


def _compute_matrices(y: np.ndarray, sr: int) -> dict:
    """Compute all librosa feature matrices from a single STFT."""
    S = np.abs(librosa.stft(y, hop_length=_HOP_LENGTH))

    chroma = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=_HOP_LENGTH)
    rms = librosa.feature.rms(S=S)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)

    np.square(S, out=S)
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr, hop_length=_HOP_LENGTH)
    del S

    mel_db = librosa.power_to_db(mel_S)
    del mel_S

    mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)
    flux = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=_HOP_LENGTH)
    del mel_db

    tempo_val, _ = librosa.beat.beat_track(onset_envelope=flux, sr=sr, hop_length=_HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=_HOP_LENGTH)

    harmonic, percussive = librosa.effects.hpss(y)
    rms_harm = librosa.feature.rms(y=harmonic, hop_length=_HOP_LENGTH)[0]
    rms_total = rms[0] + 1e-8
    harm_ratio_frames = np.clip(rms_harm / rms_total, 0.0, 1.0)

    return {
        "tempo": tempo_val, "mfcc": mfcc, "chroma": chroma, "rms": rms,
        "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
        "zcr": zcr, "flux": flux, "harmonic": harmonic,
        "percussive": percussive, "harm_ratio_frames": harm_ratio_frames,
    }


def _build_frames(mats: dict, times: np.ndarray) -> list[dict]:
    """Build frame-level feature dicts from librosa matrices."""
    mfcc = mats["mfcc"]
    chroma = mats["chroma"]
    rms = mats["rms"]
    zcr = mats["zcr"]
    centroid = mats["centroid"]
    bandwidth = mats["bandwidth"]
    rolloff = mats["rolloff"]
    flux = mats["flux"]
    harm_ratio_frames = mats["harm_ratio_frames"]

    n_frames = times.shape[0]
    frames_per_win = max(1, int(FRAME_WINDOW_S * _SR / _HOP_LENGTH))
    hop_frames = max(1, int(FRAME_HOP_S * _SR / _HOP_LENGTH))

    result: list[dict] = []
    start = 0
    while start < n_frames:
        end = min(start + frames_per_win, n_frames)
        sl = slice(start, end)
        result.append({
            "timestamp_start": float(times[start]),
            "timestamp_end": float(times[end - 1]),
            "mfcc_means": np.mean(mfcc[:, sl], axis=1).tolist(),
            "chroma_means": np.mean(chroma[:, sl], axis=1).tolist(),
            "rms_mean": float(np.mean(rms[0, sl])),
            "zcr_mean": float(np.mean(zcr[0, sl])),
            "spectral_centroid_mean": float(np.mean(centroid[0, sl])),
            "spectral_bandwidth_mean": float(np.mean(bandwidth[0, sl])),
            "spectral_rolloff_mean": float(np.mean(rolloff[0, sl])),
            "spectral_flux_mean": float(np.mean(flux[sl])),
            "harmonic_ratio": float(np.mean(harm_ratio_frames[sl])),
        })
        if end >= n_frames:
            break
        start += hop_frames
    return result


def _norm_mfcc(arr) -> list[float]:
    a = np.asarray(arr, dtype=np.float32)
    return (np.tanh(a / _MFCC_SCALES) * 0.5 + 0.5).tolist()


def _frame_to_embedding(f: dict) -> list[float]:
    """Convert a frame dict to 32-dim embedding."""
    return [
        *_norm_mfcc(f["mfcc_means"]),
        *[float(v) for v in f["chroma_means"]],
        _clamp_norm(f["rms_mean"], _RMS_MIN, _RMS_MAX),
        _clamp_norm(f["zcr_mean"], _ZCR_MIN, _ZCR_MAX),
        _clamp_norm(f["spectral_centroid_mean"], _CENTROID_MIN, _CENTROID_MAX),
        _clamp_norm(f["spectral_bandwidth_mean"], _BANDWIDTH_MIN, _BANDWIDTH_MAX),
        _clamp_norm(f["spectral_rolloff_mean"], _ROLLOFF_MIN, _ROLLOFF_MAX),
        _clamp_norm(f["spectral_flux_mean"], _FLUX_MIN, _FLUX_MAX),
        float(np.clip(f["harmonic_ratio"], 0.0, 1.0)),
    ]


def _check_auth(secret: str | None):
    if _API_SECRET and secret != _API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API secret")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ok", "service": "orpheus-extractor"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    secret: str | None = Query(default=None),
):
    """
    Upload an audio file → get full SongFeatures + FrameFeatures JSON.

    Returns:
      { "song": { ... all song-level fields ... },
        "frames": [ { ... per-frame fields + embedding ... }, ... ] }
    """
    _check_auth(secret)

    tmp_path = Path(tempfile.gettempdir()) / f"extract_{uuid.uuid4().hex}.mp3"
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
        audio_path = str(tmp_path)

        # ── VibeNet ────────────────────────────────────────────────────────
        model = _get_model()
        vibe = model.predict(audio_path)[0]

        # ── librosa ────────────────────────────────────────────────────────
        y, sr = librosa.load(audio_path, sr=_SR, duration=_MAX_DURATION_S)
        duration_s = float(y.shape[0] / sr)
        mats = _compute_matrices(y, sr)

        harmonic = mats.pop("harmonic")
        percussive = mats.pop("percussive")
        tempo_bpm = float(np.atleast_1d(mats["tempo"])[0])
        key_name, mode = _detect_key_mode(mats["chroma"])
        flux_mean = float(np.mean(mats["flux"]))
        flux_std = float(np.std(mats["flux"]))
        harm_ratio = float(np.clip(
            np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(y**2)) + 1e-8), 0, 1,
        ))
        instrument_profile = _estimate_instrument_profile(
            y, harmonic, percussive, mats["chroma"], mats["centroid"][0],
        )
        del y, harmonic, percussive
        gc.collect()

        times = librosa.times_like(mats["rms"], sr=sr, hop_length=_HOP_LENGTH)

        # ── Song-level response ────────────────────────────────────────────
        song = {
            "acousticness": vibe.acousticness,
            "danceability": vibe.danceability,
            "energy": vibe.energy,
            "instrumentalness": vibe.instrumentalness,
            "liveness": vibe.liveness,
            "speechiness": vibe.speechiness,
            "valence": vibe.valence,
            "tempo_bpm": tempo_bpm,
            "key": key_name,
            "mode": mode,
            "rms_mean": float(np.mean(mats["rms"])),
            "spectral_centroid_mean": float(np.mean(mats["centroid"])),
            "spectral_bandwidth_mean": float(np.mean(mats["bandwidth"])),
            "spectral_rolloff_mean": float(np.mean(mats["rolloff"])),
            "zcr_mean": float(np.mean(mats["zcr"])),
            "spectral_flux_mean": flux_mean,
            "spectral_flux_std": flux_std,
            "harmonic_ratio": harm_ratio,
            "mfcc_means": np.mean(mats["mfcc"], axis=1).tolist(),
            "instrument_profile": instrument_profile,
            "duration_s": duration_s,
        }

        # ── Frame-level response ──────────────────────────────────────────
        frames_raw = _build_frames(mats, times)
        frames = []
        for f in frames_raw:
            f["embedding"] = _frame_to_embedding(f)
            frames.append(f)

        logger.info("Extracted: %d frames, duration=%.1fs", len(frames), duration_s)
        return {"song": song, "frames": frames}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Extraction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        gc.collect()


@app.post("/extract-snippet")
async def extract_snippet(
    file: UploadFile = File(...),
    secret: str | None = Query(default=None),
):
    """
    Upload a short audio clip → get a 32-dim snippet embedding.

    Returns: { "embedding": [ ... 32 floats ... ] }
    """
    _check_auth(secret)

    tmp_path = Path(tempfile.gettempdir()) / f"snippet_{uuid.uuid4().hex}.mp3"
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
        audio_path = str(tmp_path)

        y, sr = librosa.load(audio_path, sr=_SR, duration=_MAX_DURATION_S)
        mats = _compute_matrices(y, sr)

        harmonic = mats.pop("harmonic")
        harm_ratio = float(np.clip(
            np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(y**2)) + 1e-8), 0, 1,
        ))
        del y, harmonic
        mats.pop("percussive", None)

        frame = {
            "mfcc_means": np.mean(mats["mfcc"], axis=1).tolist(),
            "chroma_means": np.mean(mats["chroma"], axis=1).tolist(),
            "rms_mean": float(np.mean(mats["rms"])),
            "zcr_mean": float(np.mean(mats["zcr"])),
            "spectral_centroid_mean": float(np.mean(mats["centroid"])),
            "spectral_bandwidth_mean": float(np.mean(mats["bandwidth"])),
            "spectral_rolloff_mean": float(np.mean(mats["rolloff"])),
            "spectral_flux_mean": float(np.mean(mats["flux"])),
            "harmonic_ratio": harm_ratio,
        }

        return {"embedding": _frame_to_embedding(frame)}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Snippet extraction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        gc.collect()
