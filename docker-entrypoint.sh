#!/bin/sh
set -eu

exec python -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-7860}"
