# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System utilities and FFmpeg (as fallback, though PyAV bundles its own codec libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Scratch directory for optional in-flight temporary buffers
RUN mkdir -p /app/scratch

EXPOSE 8000

# PORT is injected at runtime by Heroku / DO App Platform.
# Falls back to 8000 for local Docker usage.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
