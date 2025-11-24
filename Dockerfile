# ==========================================
# Multi-Stage Dockerfile for RAG Chatbot
# ==========================================

# ---------- Stage 1: Builder ----------
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.10-slim

# Metadata
LABEL maintainer="pravallipasala@gmail.com"
LABEL description="RAG Chatbot API with FastAPI"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH to include user site-packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY --chown=appuser:appuser app /app/app

# Create necessary directories
RUN mkdir -p /app/artifacts /app/data /app/logs && \
    chown -R appuser:appuser /app

# Copy additional files
COPY --chown=appuser:appuser run_me_once.py /app/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Build and Run Instructions:
# ==========================================
#
# Build:
#   docker build -t rag-chatbot:latest .
#
# Run with environment variables:
#   docker run -d \
#     -p 8000:8000 \
#     -v $(pwd)/artifacts:/app/artifacts \
#     -v $(pwd)/data:/app/data \
#     -e OPENAI_API_KEY=your_key_here \
#     --name rag-chatbot \
#     rag-chatbot:latest
#
# Run with .env file (development):
#   docker run -d \
#     -p 8000:8000 \
#     -v $(pwd)/artifacts:/app/artifacts \
#     -v $(pwd)/data:/app/data \
#     --env-file .env \
#     --name rag-chatbot \
#     rag-chatbot:latest
#
# ==========================================
