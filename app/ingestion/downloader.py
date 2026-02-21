"""
YouTube audio downloader built on yt-dlp.

Provides two operations:
  - search(query)            → list of VideoMeta (no download)
  - download(video_id, dest) → local mp3 path + VideoMeta
"""

from __future__ import annotations

import asyncio
import base64
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


# YouTube client rotation: android/ios bypass the "sign in to confirm" block
# that Heroku/cloud datacenter IPs trigger on the web client.
_YT_PLAYER_CLIENTS = ["android", "ios", "web"]

# Cached path to the decoded cookie file (None = not yet resolved)
_cookie_file: str | None = None
_cookie_resolved: bool = False


def _get_cookie_file() -> str | None:
    """
    Decode $YT_COOKIES_B64 (base64-encoded cookies.txt) to /tmp/yt_cookies.txt
    on first call and return the path.  Returns None if the env var is unset.

    To set up:
      1. Sign in to YouTube in your browser.
      2. Export cookies.txt with a browser extension (Netscape format).
      3. Base64-encode: python -c "import base64,sys; sys.stdout.write(base64.b64encode(open('cookies.txt','rb').read()).decode())"
      4. heroku config:set YT_COOKIES_B64='<output>' --app orpheus-api
    """
    global _cookie_file, _cookie_resolved
    if _cookie_resolved:
        return _cookie_file
    _cookie_resolved = True
    raw = os.getenv("YT_COOKIES_B64", "").strip()
    if not raw:
        logger.debug("YT_COOKIES_B64 not set; running without YouTube cookies.")
        return None
    try:
        cookie_path = "/tmp/yt_cookies.txt"
        decoded = base64.b64decode(raw)
        if len(decoded) < 20 or not decoded.lstrip().startswith(b"# Netscape"):
            logger.warning(
                "YT_COOKIES_B64 decoded to %d bytes but does not look like a "
                "Netscape cookie file — ignoring. Re-export and re-set it.",
                len(decoded),
            )
            return None
        Path(cookie_path).write_bytes(decoded)
        logger.info(
            "YouTube cookies written to %s (%d bytes)", cookie_path, len(decoded)
        )
        _cookie_file = cookie_path
    except Exception as exc:
        logger.warning("Failed to decode YT_COOKIES_B64: %s", exc)
    return _cookie_file


def _base_ydl_opts() -> dict:
    """Common yt-dlp options shared by search and download."""
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Use android client first — avoids bot-check on datacenter IPs.
        # yt-dlp tries each client in order until one succeeds.
        "extractor_args": {
            "youtube": {
                "player_client": _YT_PLAYER_CLIENTS,
            }
        },
        # Small polite delay between requests
        "sleep_interval_requests": 15,
    }
    cookie_file = _get_cookie_file()
    if cookie_file:
        opts["cookiefile"] = cookie_file
    return opts


def _build_ydl_opts(output_template: str) -> dict:
    """yt-dlp options for audio-only mp3 download."""
    opts = _base_ydl_opts()
    opts.update(
        {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": output_template,
            # Skip videos longer than 10 minutes (likely not songs)
            "match_filter": yt_dlp.utils.match_filter_func("duration < 600"),
        }
    )
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
    Returns up to max_results VideoMeta objects, or [] on any error.
    """
    search_query = f"ytsearch{max_results}:{query}"
    opts = _base_ydl_opts()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("YouTube search failed (will skip): %s", exc)
        return []
    except Exception as exc:
        logger.warning("YouTube search unexpected error (will skip): %s", exc)
        return []
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

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"YouTube download blocked/failed for {video_id}: {exc}"
        ) from exc

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
