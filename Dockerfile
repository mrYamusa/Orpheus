# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ffmpeg – required by yt-dlp for audio post-processing
# deno   – yt-dlp's recommended JS runtime for YouTube EJS challenge solving
#          (enabled by default, no --js-runtimes flag needed)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "
import urllib.request, zipfile, os
url = 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip'
urllib.request.urlretrieve(url, '/tmp/deno.zip')
with zipfile.ZipFile('/tmp/deno.zip') as z:
    z.extractall('/usr/local/bin/')
os.chmod('/usr/local/bin/deno', 0o755)
os.remove('/tmp/deno.zip')" \
    && deno --version

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
