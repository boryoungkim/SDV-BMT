"""Scope gate (S1A-12, updated for the authorized Stage 1R-G GPU track).

Stage 1R-G adds exactly the native measurement plane (``native/``, C++), its
Python client (``dut/native.py``), the realtime run-clock module
(``runner/realtime.py``), the NVIDIA reference-accelerator adapter
(``dut/nvidia``) and device-dependent tests (``tests/device``).  Analysis/
statistics, telemetry services, System-level code, plugins, threads/async and
new Python dependencies remain out of scope.
"""

from __future__ import annotations

import re

from tests._helpers import REPO_ROOT, SRC_ROOT

FORBIDDEN_PATHS = [
    SRC_ROOT / "analysis",
    SRC_ROOT / "telemetry",
    SRC_ROOT / "system",
    SRC_ROOT / "system_impact",
    SRC_ROOT / "safety",
    SRC_ROOT / "autoware",
    SRC_ROOT / "ros2",
    SRC_ROOT / "integrations",
    SRC_ROOT / "plugins",
    REPO_ROOT / "integrations",
]

# Stage 1R-G additions that must exist (the scope change is explicit, not accidental).
STAGE_1RG_PATHS = [
    SRC_ROOT / "runner" / "realtime.py",
    SRC_ROOT / "dut" / "native.py",
    SRC_ROOT / "dut" / "nvidia" / "adapter.py",
    REPO_ROOT / "native" / "src" / "common" / "loop.hpp",
    REPO_ROOT / "native" / "src" / "nvidia" / "backend.hpp",
    REPO_ROOT / "native" / "src" / "fake" / "backend.hpp",
]

# Identifiers that would indicate analysis/statistics or realtime/async code paths.
FORBIDDEN_IDENTIFIER_PATTERN = re.compile(
    r"\b(mann_?whitney|holm|bootstrap|p_?value|degradation_?analy[sz]er|"
    r"submit_async|async\s+def|await\s|callback_queue|ThreadPool|ProcessPool|"
    r"entry_points)\b",
    re.IGNORECASE,
)


def test_forbidden_paths_do_not_exist():
    present = [str(p.relative_to(REPO_ROOT)) for p in FORBIDDEN_PATHS if p.exists()]
    assert present == []


def test_stage_1rg_scope_is_exactly_the_authorized_additions():
    assert [str(p.relative_to(REPO_ROOT)) for p in STAGE_1RG_PATHS if not p.exists()] == []
    nvidia = sorted(p.name for p in (SRC_ROOT / "dut" / "nvidia").glob("*.py"))
    assert nvidia == ["__init__.py", "adapter.py"]


def test_no_analysis_or_async_identifiers_in_source():
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN_IDENTIFIER_PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []


def _toml_list(text: str, key: str) -> list[str]:
    match = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    assert match, key
    return re.findall(r'"([^"]+)"', match.group(1))


def test_declared_dependencies_are_exactly_the_approved_set():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert _toml_list(text, "dependencies") == ["PyYAML>=6.0"]
    assert _toml_list(text, "dev") == ["pytest>=7"]
    assert _toml_list(text, "requires") == ["setuptools>=70.1"]
    assert 'build-backend = "setuptools.build_meta"' in text
    assert "entry-points" not in text.replace("[project.scripts]", "")


def test_cli_has_no_analyze_command():
    cli = (SRC_ROOT / "cli" / "main.py")
    if cli.exists():
        assert '"analyze"' not in cli.read_text(encoding="utf-8")
