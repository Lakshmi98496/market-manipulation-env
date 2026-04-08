FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY server/ ./server/
COPY tasks/  ./tasks/
COPY openenv.yaml .

# Make packages importable
RUN touch server/__init__.py tasks/__init__.py

# HF Spaces runs on port 7860
EXPOSE 7860

# Health check (required by pre-validation)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.env:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
