"""Tests for dynamic port fallback and base URL discovery."""

from __future__ import annotations

import socket

import pytest

from sre_env.client import SREEnv
from sre_env.utils import port_resolver


def _bind_busy_port(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def test_resolve_server_port_prefers_explicit_port_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9010")

    resolved = port_resolver.resolve_server_port(host="127.0.0.1", preferred_port=7860)

    assert resolved == 9010


def test_resolve_server_port_falls_back_when_preferred_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORT", raising=False)
    busy_sock = _bind_busy_port("127.0.0.1", 7860)
    try:
        resolved = port_resolver.resolve_server_port(host="127.0.0.1", preferred_port=7860)
    finally:
        busy_sock.close()

    assert resolved != 7860
    assert 1 <= resolved <= 65535


def test_port_file_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    port_file = tmp_path / ".openenv_port"
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_file))

    port_resolver.write_selected_port(8123)
    assert port_resolver.read_selected_port() == 8123


def test_port_file_path_prefers_openenv_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("OPENENV_PORT_FILE", raising=False)
    monkeypatch.setenv("OPENENV_REPO_ROOT", str(tmp_path))

    assert port_resolver.port_file_path() == tmp_path / ".openenv_port"


def test_write_selected_port_permission_error_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    port_dir = tmp_path / ".openenv_port"
    port_dir.mkdir()
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_dir))

    port_resolver.write_selected_port(8123)
    captured = capsys.readouterr()
    assert "[OPENENV][WARN] Failed to write port file" in captured.out


def test_client_base_url_priority_explicit_over_env_and_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    port_file = tmp_path / ".openenv_port"
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_file))
    monkeypatch.setenv("OPENENV_BASE_URL", "http://127.0.0.1:9999")
    port_resolver.write_selected_port(8123)

    client = SREEnv(base_url="http://127.0.0.1:7000")
    try:
        assert client.base_url == "http://127.0.0.1:7000"
    finally:
        import asyncio

        asyncio.run(client.close())


def test_client_base_url_uses_env_then_file_then_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    port_file = tmp_path / ".openenv_port"
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_file))

    client = SREEnv()
    try:
        assert client.base_url == "http://127.0.0.1:7860"
    finally:
        import asyncio

        asyncio.run(client.close())

    port_resolver.write_selected_port(8123)
    client = SREEnv()
    try:
        assert client.base_url == "http://127.0.0.1:8123"
    finally:
        import asyncio

        asyncio.run(client.close())

    monkeypatch.setenv("OPENENV_BASE_URL", "http://127.0.0.1:9555")
    client = SREEnv()
    try:
        assert client.base_url == "http://127.0.0.1:9555"
    finally:
        import asyncio

        asyncio.run(client.close())
