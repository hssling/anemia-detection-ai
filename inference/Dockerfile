# inference/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY inference/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy training package (needed for model classes)
COPY training/ training/
COPY inference/ inference/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-7860}/health" || exit 1'

CMD ["sh", "-c", "uvicorn inference.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
