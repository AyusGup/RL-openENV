"""Pytest fixtures shared across the repo test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Use pytest temp dirs when available, with a local fallback if needed."""
    if os.getenv("OPENENV_TEST_LOCAL_TMP") != "1":
        try:
            path = tmp_path_factory.mktemp("case")
            yield path
            return
        except PermissionError:
            pass

    base_dir = Path(".test-tmp")
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / f"case-{uuid4().hex}"
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
