# inference/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

COPY inference/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ training/
COPY inference/ inference/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-7860}/health" || exit 1'

CMD ["sh", "-c", "uvicorn inference.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
