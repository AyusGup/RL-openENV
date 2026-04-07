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


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    OPENENV_REPO_ROOT=/app/sre_env

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app --no-create-home appuser \
    && mkdir -p /app/sre_env/workspace

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --chown=appuser:appgroup . /app/sre_env

RUN chmod +x /app/sre_env/docker-entrypoint.sh

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-7860}/health')"

ENTRYPOINT ["/app/sre_env/docker-entrypoint.sh"]
