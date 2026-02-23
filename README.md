# Orpheus

Voice-activated music discovery engine. Ingests songs from Spotify / YouTube, extracts audio features using deep learning, stores them as vectors in Qdrant, and serves semantic playlist search.

## Architecture

Orpheus is split into two services:

| Service | Runs on | Purpose |
|---|---|---|
| **Orpheus API** | Heroku (512 MB) | Handles search, ingestion scheduling, YouTube downloads, and Qdrant storage |
| **Orpheus Extractor** | Hugging Face Space (16 GB) | Runs the heavy ML inference (VibeNet + librosa) and returns feature vectors via HTTP |

```
┌────────────────────────────────────────────────────────────────────────┐
│  Client (voice app / curl / Scalar UI)                                │
│    POST /search          → semantic playlist search                   │
│    POST /match/snippet   → Shazam-style audio recognition             │
│    POST /scheduler/trigger → kick off an ingestion cycle              │
└──────────────────┬─────────────────────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │   Orpheus API   │  (Heroku)
          │   FastAPI        │
          │                  │
          │  ┌────────────┐  │       ┌──────────────────┐
          │  │ Downloader │──│──────►│  YouTube (yt-dlp) │
          │  └─────┬──────┘  │       └──────────────────┘
          │        │ mp3     │
          │  ┌─────▼──────┐  │  HTTP  ┌─────────────────────┐
          │  │ Extractor  │──│──────►│  HF Space Extractor  │
          │  │  (proxy)   │◄─│───────│  VibeNet + librosa   │
          │  └─────┬──────┘  │  JSON  └─────────────────────┘
          │        │ vectors │
          │  ┌─────▼──────┐  │       ┌──────────────────┐
          │  │  Qdrant    │──│──────►│  Qdrant Cloud    │
          │  │  client    │  │       │  (vector DB)     │
          │  └────────────┘  │       └──────────────────┘
          │                  │
          │  ┌────────────┐  │       ┌──────────────────┐
          │  │ Spotify    │──│──────►│  Spotify API     │
          │  │  client    │  │       │  (song metadata) │
          │  └────────────┘  │       └──────────────────┘
          └──────────────────┘
```

## Ingestion Pipeline

Each ingestion cycle runs these steps per song (sequentially, one at a time to stay within memory limits):

1. **Discover** — Spotify client picks a category (afro, pop, rnb, etc.) and searches for tracks. Falls back to a YouTube query pool if Spotify credentials aren't set.
2. **De-duplicate** — Checks Qdrant to skip songs already in the library.
3. **Download** — `yt-dlp` downloads the audio as MP3 from YouTube (with cookie auth and Deno JS runtime for cipher solving).
4. **Extract** — The MP3 is uploaded to the HF Space extractor, which returns:
   - **Song-level features** (30-dim): 7 VibeNet mood scores + 10 librosa spectral scalars + 13 MFCCs
   - **Frame-level features** (32-dim): per-window embeddings for snippet matching (5s windows, 2.5s hop)
5. **Store** — Song embedding + metadata → `orpheus_songs` collection; frame embeddings → `orpheus_frames` collection (both in Qdrant).
6. **Cleanup** — Temporary audio files are deleted and garbage collected.

The scheduler runs this cycle every 30 minutes (configurable), processing 4 songs per run.

## Embedding Schema

### Song Embedding (30 dimensions)

| Dims | Source | Features |
|---|---|---|
| 0–6 | VibeNet (ONNX) | acousticness, danceability, energy, instrumentalness, liveness, speechiness, valence |
| 7–9 | librosa | tempo (normalised), key (0–11 / 11), mode (major=1 / minor=0) |
| 10–16 | librosa | rms, spectral centroid, bandwidth, rolloff, zcr, spectral flux, harmonic ratio |
| 17–29 | librosa | 13 MFCCs (tanh-squashed to [0, 1]) |

### Frame Embedding (32 dimensions)

| Dims | Source | Features |
|---|---|---|
| 0–12 | librosa | 13 MFCCs |
| 13–24 | librosa | 12 chroma bins |
| 25–31 | librosa | rms, zcr, spectral centroid, bandwidth, rolloff, flux, harmonic ratio |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Qdrant collection statistics (song count, frame count) |
| `GET` | `/scheduler/status` | Next scheduled run, song source, config |
| `POST` | `/scheduler/trigger` | Manually fire an ingestion cycle |
| `POST` | `/ingest/song` | Ingest a specific YouTube video by ID |
| `POST` | `/search` | Semantic song search from structured JSON query |
| `POST` | `/match/snippet` | Upload an audio clip for Shazam-style matching |
| `GET` | `/scalar` | Interactive API docs (Scalar UI) |

### Search Request Example

```json
{
  "energy": 0.8,
  "valence": 0.6,
  "danceability": 0.7,
  "tempo_min": 120,
  "tempo_max": 150,
  "mode": "major",
  "limit": 10
}
```

All vibe fields (`energy`, `valence`, etc.) are floats in `[0, 1]`. Omit any field to leave it unconstrained (defaults to 0.5 in the query vector). `tempo_min` / `tempo_max` are in BPM and applied as hard Qdrant payload filters.

## Project Structure

```
Orpheus/
├── app/
│   ├── main.py              # FastAPI app factory, lifespan hooks
│   ├── config.py            # All settings via env vars
│   ├── api/
│   │   └── routes.py        # HTTP endpoint definitions
│   ├── database/
│   │   └── qdrant.py        # Qdrant client, collection bootstrap, upsert, search
│   ├── ingestion/
│   │   ├── pipeline.py      # Orchestrates download → extract → store
│   │   ├── downloader.py    # yt-dlp wrapper (YouTube search + download)
│   │   ├── extractor.py     # HTTP proxy to HF Space (SongFeatures / FrameFeatures)
│   │   └── spotify_client.py # Spotify API integration (track discovery)
│   └── scheduler/
│       └── jobs.py          # APScheduler async job definitions
├── orpheus-extractor/        # Separate HF Space service
│   ├── app.py               # FastAPI extraction service (VibeNet + librosa)
│   ├── Dockerfile           # python:3.11-slim + ffmpeg
│   ├── requirements.txt     # Heavy ML deps (librosa, onnxruntime, vibenet)
│   └── README.md            # HF Space metadata
├── Dockerfile               # Heroku container (python:3.11-slim + ffmpeg + deno)
├── requirements.txt         # Lightweight deps (no ML libraries)
└── README.md
```

## Environment Variables

### Orpheus API (Heroku)

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | Yes | Qdrant Cloud endpoint URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `SPOTIFY_CLIENT_ID` | Yes* | Spotify app client ID |
| `SPOTIFY_CLIENT_SECRET` | Yes* | Spotify app client secret |
| `YT_COOKIES_B64` | Recommended | Base64-encoded Netscape cookie file for YouTube auth |
| `HF_EXTRACTION_URL` | No | HF Space URL (defaults to `https://mryamusa-orpheus-extractor.hf.space`) |
| `HF_EXTRACTION_SECRET` | No | Shared secret for app-level auth with the extractor |
| `HF_TOKEN` | Yes** | Hugging Face API token (required if the Space is private) |
| `SONGS_PER_RUN` | No | Songs per ingestion cycle (default: 4) |
| `SCHEDULE_MINUTES` | No | Interval between auto-ingestion runs (default: 30) |
| `PORT` | No | Server port (set automatically by Heroku) |

\* Falls back to YouTube query pool if Spotify is not configured.  
\** Only required when the HF Space is set to private visibility.

### Orpheus Extractor (HF Space)

| Variable | Required | Description |
|---|---|---|
| `API_SECRET` | No | Shared secret to authenticate incoming requests |

## Local Development

### Prerequisites

- Python 3.11+
- ffmpeg installed and on PATH
- A Qdrant instance (local Docker or cloud)

### Setup

```bash
# Clone the repo
git clone <repo-url> && cd Orpheus

# Create virtual environment
python -m venv orpheus_env
source orpheus_env/bin/activate  # or orpheus_env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # edit with your credentials

# Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at `http://localhost:8000/scalar`.

### Running with Docker

```bash
docker build -t orpheus .
docker run -p 8000:8000 --env-file .env orpheus
```

## Deployment

### Heroku (API)

The API is deployed as a Docker container on Heroku:

```bash
heroku container:push web --app orpheus-api
heroku container:release web --app orpheus-api

# Or via git push (with heroku.yml or Dockerfile auto-detection)
git push heroku main
```

Set config vars:

```bash
heroku config:set QDRANT_URL=https://... QDRANT_API_KEY=... --app orpheus-api
heroku config:set SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=... --app orpheus-api
heroku config:set HF_TOKEN=hf_... --app orpheus-api
```

### Hugging Face Space (Extractor)

The `orpheus-extractor/` directory is pushed to a separate HF Space repo:

```bash
cd orpheus-extractor
git init && git remote add origin https://huggingface.co/spaces/<user>/orpheus-extractor
git add -A && git commit -m "deploy" && git push origin main
```

The Space auto-builds from the Dockerfile and exposes the extraction API on port 7860.

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector database | Qdrant Cloud |
| Audio features | librosa, VibeNet (ONNX), NumPy |
| YouTube downloads | yt-dlp + Deno (JS runtime) + ffmpeg |
| Song discovery | Spotify Web API (via spotipy) |
| Scheduling | APScheduler (AsyncIO) |
| HTTP client | httpx (async) |
| API docs | Scalar |
| Containerisation | Docker |
| Hosting | Heroku (API) + Hugging Face Spaces (Extractor) |

## License

MIT
