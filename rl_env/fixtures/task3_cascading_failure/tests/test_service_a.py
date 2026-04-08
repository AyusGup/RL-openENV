"""Tests for Service A timeout behavior."""

from __future__ import annotations

import ast
from pathlib import Path


def extract_http_timeout_seconds() -> float:
    """Read the configured httpx timeout from service_a/main.py."""
    
    source_path = Path(__file__).resolve().parents[1] / "service_a" / "main.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "AsyncClient":
            continue
        for keyword in node.keywords:
            if keyword.arg == "timeout" and isinstance(keyword.value, ast.Constant):
                return float(keyword.value.value)

    raise AssertionError("Could not find httpx.AsyncClient(timeout=...) in service_a/main.py")


def test_service_a_timeout_budget_is_large_enough() -> None:
    timeout_seconds = extract_http_timeout_seconds()

    assert timeout_seconds >= 0.3
    assert timeout_seconds <= 1.0
