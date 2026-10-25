FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cbir/ cbir/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/cache/features data/cache/indexes

EXPOSE 8000 8501

# Default: start the API backend
CMD ["uvicorn", "cbir.api:app", "--host", "0.0.0.0", "--port", "8000"]
