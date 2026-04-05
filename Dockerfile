FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md /build/
COPY sre_env /build/sre_env
COPY server /build/server

RUN pip install .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV HOME=/app
ENV OPENENV_REPO_ROOT=/app

WORKDIR /app

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app appuser \
    && mkdir -p /app/workspace

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup openenv.yaml /app/openenv.yaml
COPY --chown=appuser:appgroup sre_env /app/sre_env
COPY --chown=appuser:appgroup server /app/server
COPY --chown=appuser:appgroup fixtures /app/fixtures

USER appuser

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
