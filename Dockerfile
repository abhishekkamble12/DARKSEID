# =============================================================================
# Dockerfile for Supervisor Multi-Agent System with RAG
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
# - libpq-dev, gcc: PostgreSQL driver
# - poppler-utils: PDF processing
# - tesseract-ocr: OCR for images in PDFs
# - libmagic1: File type detection
# - ffmpeg, libsm6, libxext6: OpenCV dependencies (for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY app/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code
COPY app/ .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port (if running as API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run in interactive mode with Docker flag
CMD ["python", "project.py", "--docker"]
