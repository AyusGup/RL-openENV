"""Port resolution and OpenEnv base URL helpers."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Optional

DEFAULT_OPENENV_PORT = 7860
DEFAULT_CLIENT_HOST = "127.0.0.1"
DEFAULT_SERVER_HOST = "0.0.0.0"
PORT_FILE_NAME = ".openenv_port"


def repo_root() -> Path:
    """Return repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def port_file_path() -> Path:
    """Return the deterministic file path used to store selected port."""
    override = os.getenv("OPENENV_PORT_FILE")
    if override:
        return Path(override)
    return repo_root() / PORT_FILE_NAME


def parse_port(value: str | None) -> Optional[int]:
    """Parse and validate a TCP port number."""
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if 1 <= parsed <= 65535:
        return parsed
    return None


def is_port_available(host: str, port: int) -> bool:
    """Return whether host:port can be bound."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def choose_ephemeral_port(host: str) -> int:
    """Ask the OS for a free port and return it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def resolve_server_port(
    host: str = DEFAULT_SERVER_HOST,
    preferred_port: int = DEFAULT_OPENENV_PORT,
) -> int:
    """Resolve server bind port with explicit PORT override and fallback."""
    env_port = parse_port(os.getenv("PORT"))
    if env_port is not None:
        return env_port

    if is_port_available(host, preferred_port):
        return preferred_port
    return choose_ephemeral_port(host)


def write_selected_port(port: int) -> None:
    """Persist selected server port for local client auto-discovery."""
    path = port_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(port), encoding="utf-8")


def read_selected_port() -> Optional[int]:
    """Read selected server port from deterministic file."""
    path = port_file_path()
    if not path.exists():
        return None
    return parse_port(path.read_text(encoding="utf-8").strip())


def build_local_base_url(port: int, host: str = DEFAULT_CLIENT_HOST) -> str:
    """Build a local OpenEnv base URL for HTTP clients."""
    return f"http://{host}:{port}"


def resolve_base_url(explicit_base_url: str | None = None) -> str:
    """Resolve base URL via explicit arg, env var, port file, then default."""
    if explicit_base_url:
        return explicit_base_url.rstrip("/")

    env_url = (os.getenv("OPENENV_BASE_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")

    file_port = read_selected_port()
    if file_port is not None:
        return build_local_base_url(file_port)

    return build_local_base_url(DEFAULT_OPENENV_PORT)
