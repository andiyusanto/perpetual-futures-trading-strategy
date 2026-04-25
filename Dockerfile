# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System build deps (needed for some ccxt / numpy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config/ config/
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -e .


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN addgroup --system apfts && adduser --system --ingroup apfts apfts

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY --chown=apfts:apfts config/ config/
COPY --chown=apfts:apfts src/ src/

# Runtime data + log directories (mounted as volumes in production)
RUN mkdir -p /app/data /app/logs && chown -R apfts:apfts /app

USER apfts

# Health check — confirms the Python environment is intact
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from src.strategy.engine import V3StrategyEngine; print('ok')" || exit 1

# Default: single-symbol bot
# Override CMD in docker-compose to run apfts-multi-bot
CMD ["apfts-bot"]
