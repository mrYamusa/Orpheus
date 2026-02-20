"""
YouTube audio downloader built on yt-dlp.

Provides two operations:
  - search(query)            → list of VideoMeta (no download)
  - download(video_id, dest) → local mp3 path + VideoMeta
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    id: str
    title: str
    artist: str  # uploader name (best we can do without MusicBrainz)
    duration_s: float
    webpage_url: str


def _build_ydl_opts(output_template: str) -> dict:
    """Base yt-dlp options for audio-only mp3 download."""
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Skip videos longer than 10 minutes (likely not songs)
        "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
    }
    # Allow users to override ffmpeg location via env var
    ffmpeg = os.getenv("FFMPEG_PATH")
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    return opts


def _info_to_meta(info: dict) -> VideoMeta:
    return VideoMeta(
        id=info.get("id", ""),
        title=info.get("track") or info.get("title", "Unknown Title"),
        artist=info.get("artist") or info.get("uploader", "Unknown Artist"),
        duration_s=float(info.get("duration") or 0),
        webpage_url=info.get("webpage_url", ""),
    )


def _yt_search_sync(query: str, max_results: int) -> list[VideoMeta]:
    """
    Run a YouTube search without downloading anything.
    Returns up to max_results VideoMeta objects.
    """
    search_query = f"ytsearch{max_results}:{query}"
    with yt_dlp.YoutubeDL(
        {"quiet": True, "no_warnings": True, "noplaylist": True}
    ) as ydl:
        info = ydl.extract_info(search_query, download=False)
    entries = (info or {}).get("entries") or []
    results = []
    for entry in entries:
        if entry and entry.get("id"):
            results.append(_info_to_meta(entry))
    return results


def _yt_download_sync(video_id: str, dest_dir: Path) -> tuple[Path, VideoMeta]:
    """
    Download a single video as mp3 into dest_dir.
    Returns (mp3_path, VideoMeta).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(dest_dir / f"{video_id}.%(ext)s")
    opts = _build_ydl_opts(output_template)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    meta = _info_to_meta(info or {})
    mp3_path = dest_dir / f"{video_id}.mp3"

    if not mp3_path.exists():
        # yt-dlp may have named it differently; find the first mp3 in dest_dir
        candidates = list(dest_dir.glob(f"{video_id}*.mp3"))
        if not candidates:
            raise FileNotFoundError(
                f"Download produced no mp3 for {video_id} in {dest_dir}"
            )
        mp3_path = candidates[0]

    logger.info("Downloaded: %s → %s", video_id, mp3_path)
    return mp3_path, meta


# --- Async wrappers ---


async def search_youtube(query: str, max_results: int = 5) -> list[VideoMeta]:
    """Async wrapper around _yt_search_sync."""
    return await asyncio.to_thread(_yt_search_sync, query, max_results)


async def download_song(video_id: str, dest_dir: Path) -> tuple[Path, VideoMeta]:
    """Async wrapper around _yt_download_sync."""
    return await asyncio.to_thread(_yt_download_sync, video_id, dest_dir)
