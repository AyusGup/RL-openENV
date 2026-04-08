#!/bin/sh
set -eu

exec python -m uvicorn sre_env.server.app:app --host 0.0.0.0 --port "${PORT:-8000}"
