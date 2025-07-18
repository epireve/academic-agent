# Production Docker Image for Academic Agent
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
COPY academic-agent-v2/pyproject.toml academic-agent-v2/poetry.lock ./

# Install Python dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Create non-root user
RUN groupadd -r academic && useradd -r -g academic -d /app -s /sbin/nologin academic

# Copy application code
COPY academic-agent-v2/src ./src
COPY agents ./agents
COPY config ./config
COPY processors ./processors
COPY tools ./tools
COPY monitoring ./monitoring

# Copy additional dependencies
COPY requirements.txt ./
RUN poetry run pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p logs input output processed metadata tmp && \
    chown -R academic:academic /app

# Copy deployment scripts
COPY deployment/scripts/*.sh ./scripts/
RUN chmod +x ./scripts/*.sh

# Health check script
COPY deployment/healthcheck.py ./
RUN chmod +x healthcheck.py

# Switch to non-root user
USER academic

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Default command
CMD ["poetry", "run", "python", "-m", "agents.academic.main_agent", "--config", "/app/config/production.yaml"]