# Askme MCP Server — production Docker image
# Usage:
#   docker build -t askme:4.0 .
#   docker run --env-file .env -p 8080:8080 askme:4.0

FROM python:3.13-slim AS base

WORKDIR /app

# System dependencies (portaudio for sounddevice, even if voice is off)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code and default config
COPY askme/ askme/
COPY config.yaml .

# Default: voice disabled in containers (no audio hardware)
ENV ASKME_FEATURE_VOICE=0

EXPOSE 8080

ENTRYPOINT ["python", "-m", "askme"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]
