"""OpenEnv-facing server entrypoint."""

from __future__ import annotations

import uvicorn

from sre_env.server.app import app


def main() -> None:
    """Run the FastAPI application."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
