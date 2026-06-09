FROM python:3.11-slim

WORKDIR /app

# Build deps for compiled packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache directory writable by HF Spaces' non-root user
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Install Python deps first (Docker layer cache speeds up later builds)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy everything else
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "7860"]