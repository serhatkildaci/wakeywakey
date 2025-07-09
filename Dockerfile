# Multi-stage Dockerfile for WakeyWakey
# Supports both development and production deployments

# Build stage
FROM python:3.8-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

LABEL maintainer="Serhat Kildaci <serhat.kildaci@example.com>"
LABEL description="WakeyWakey - Lightweight Wake Word Detection"
LABEL version="0.1.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    portaudio19-dev \
    python3-dev \
    libasound2-dev \
    libportaudio2 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.8-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash wakeywakey
USER wakeywakey
WORKDIR /home/wakeywakey

# Copy built package from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /home/wakeywakey/app

# Set environment variables
ENV PYTHONPATH=/home/wakeywakey/app
ENV WAKEYWAKEY_CONFIG_DIR=/home/wakeywakey/.wakeywakey
ENV WAKEYWAKEY_MODEL_DIR=/home/wakeywakey/models
ENV WAKEYWAKEY_DATA_DIR=/home/wakeywakey/data

# Create necessary directories
RUN mkdir -p $WAKEYWAKEY_CONFIG_DIR $WAKEYWAKEY_MODEL_DIR $WAKEYWAKEY_DATA_DIR

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wakeywakey list-devices || exit 1

# Default command
CMD ["wakeywakey", "--help"]

# Development stage (optional)
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Keep as root for development flexibility
WORKDIR /app

# Expose additional ports for development
EXPOSE 8080 8888 6006

# Development command
CMD ["bash"]

# Minimal stage for embedded systems
FROM python:3.8-alpine as minimal

# Install minimal runtime dependencies
RUN apk add --no-cache \
    portaudio-dev \
    alsa-lib-dev \
    gcc \
    musl-dev \
    linux-headers

# Install only essential Python packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy librosa sounddevice

# Copy only essential files
COPY wakeywakey /app/wakeywakey
COPY setup.py requirements.txt /app/

WORKDIR /app
RUN pip install --no-cache-dir -e .

# Minimal command
CMD ["python", "-m", "wakeywakey.cli.main"]

# ARM/Raspberry Pi stage
FROM arm32v7/python:3.8-slim as arm

# Install ARM-specific dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY . /app
WORKDIR /app

# Install with ARM optimizations
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# Optimize for Raspberry Pi
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["wakeywakey", "detect", "--model", "/models/lightweight_model.pth"] 