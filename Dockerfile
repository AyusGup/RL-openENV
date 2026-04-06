FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY uv.lock pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY sre_env ./sre_env
COPY server ./server
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    OPENENV_REPO_ROOT=/app

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/workspace

# Copy only the built venv plus runtime assets. The application code is
# already installed into the venv via `uv sync --no-editable` in the builder.
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup openenv.yaml /app/openenv.yaml
COPY --chown=appuser:appgroup docker-entrypoint.sh /app/docker-entrypoint.sh
COPY --chown=appuser:appgroup fixtures /app/fixtures

RUN chmod +x /app/docker-entrypoint.sh

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-7860}/health')"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
