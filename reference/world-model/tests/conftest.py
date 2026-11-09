"""
Pytest configuration and test isolation fixtures for wm-control.

Enforces:
- Section 6.3 of agents.md: Strict Test Output Isolation
- Tests must NEVER write to root outputs/
- All test outputs must be written exclusively inside tests/outputs/
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

from wm_control.config import TESTS_OUTPUT_DIR


@pytest.fixture(scope="session", autouse=True)
def ensure_test_output_isolation() -> Generator[Path, None, None]:
    """
    Session-level fixture ensuring tests/outputs/ exists and root outputs/ is protected.
    """
    TESTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield TESTS_OUTPUT_DIR


@pytest.fixture
def isolated_test_dir(tmp_path: Path) -> Path:
    """
    Fixture providing an isolated subdirectory within tests/outputs/ for test runs.
    """
    test_run_dir = TESTS_OUTPUT_DIR / f"test_run_{os.getpid()}_{tmp_path.name}"
    test_run_dir.mkdir(parents=True, exist_ok=True)
    yield test_run_dir
    # Cleanup after test completes
    if test_run_dir.exists():
        shutil.rmtree(test_run_dir, ignore_errors=True)
