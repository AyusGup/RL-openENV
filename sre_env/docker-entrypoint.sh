#!/bin/sh
set -eu

export PORT="${PORT:-7860}"
exec python -m sre_env.server.app
