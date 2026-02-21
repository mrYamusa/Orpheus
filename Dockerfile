# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ffmpeg  – required by yt-dlp for audio post-processing
# nodejs  – required by yt-dlp (2025+) to decode YouTube cipher signatures
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create scratch directory
RUN mkdir -p /app/scratch

EXPOSE 8000

# PORT is injected at runtime by Heroku / DO App Platform.
# Falls back to 8000 for local Docker usage.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
