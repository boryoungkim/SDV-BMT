"""Pytest configuration shared by all test suites."""

from __future__ import annotations

import os
import subprocess

import pytest

from tests._helpers import REPO_ROOT


@pytest.fixture(scope="session")
def native_build():
    """Build the native measurement plane once per session (system toolchain; see native/Makefile).

    ``SDV_BMT_MAKE_TARGETS`` selects make targets (default: all).  Hosted CI without CUDA uses ``portable``
    (fake runner + C++ unit tests); the NVIDIA device tests then report NOT RUN (skipped), never passed."""
    targets = os.environ.get("SDV_BMT_MAKE_TARGETS", "").split()
    proc = subprocess.run(["make", "-s", "-C", str(REPO_ROOT / "native"), *targets], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"native build failed:\n{proc.stdout}\n{proc.stderr}")
    return REPO_ROOT / "native" / "build"
