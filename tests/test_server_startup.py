"""Startup behavior tests for server dynamic port selection."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

import sre_env.server.app as app_module


def _bind_busy_port(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def test_main_uses_fallback_port_and_writes_port_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    busy_sock = _bind_busy_port("127.0.0.1", 7860)
    captured: dict[str, int | str] = {}
    port_file = tmp_path / ".openenv_port"
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_file))
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.delenv("PORT", raising=False)

    def fake_uvicorn_run(_app, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(app_module.uvicorn, "run", fake_uvicorn_run)
    try:
        app_module.main()
    finally:
        busy_sock.close()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] != 7860
    assert int(port_file.read_text(encoding="utf-8").strip()) == int(captured["port"])


def test_main_honors_explicit_port_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, int | str] = {}
    port_file = tmp_path / ".openenv_port"
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_file))
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8899")

    def fake_uvicorn_run(_app, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(app_module.uvicorn, "run", fake_uvicorn_run)
    app_module.main()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8899
    assert int(port_file.read_text(encoding="utf-8").strip()) == 8899


def test_main_continues_when_port_file_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, int | str] = {}
    port_dir = tmp_path / ".openenv_port"
    port_dir.mkdir()
    monkeypatch.setenv("OPENENV_PORT_FILE", str(port_dir))
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "7860")

    def fake_uvicorn_run(_app, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(app_module.uvicorn, "run", fake_uvicorn_run)
    app_module.main()
    out = capsys.readouterr().out

    assert "[OPENENV][WARN] Failed to write port file" in out
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 7860
