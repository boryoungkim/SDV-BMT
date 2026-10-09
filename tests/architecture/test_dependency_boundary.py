"""Architecture gate (S1A-10): package dependency directions and forbidden imports.

Allowlist-based: every import in ``src/sdv_bmt`` must be either a permitted
standard-library module, PyYAML inside ``runner/plan.py`` only, or an
``sdv_bmt`` subpackage allowed by the layer matrix below.

Stage 1R-G (authorized GPU track, approved architecture "C++ measurement plane,
Python analysis plane") allows exactly these stdlib exceptions: ``time`` (host
monotonic clock) in the realtime run-clock module and in the native-plane
client, and ``subprocess`` in the native-plane client (it starts the native
runner process).  No ``ctypes``: vendor libraries are used only by the native
runner (``native/``), never by the Python package.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from tests._helpers import SRC_ROOT

PACKAGE = "sdv_bmt"

# subpackage -> sdv_bmt subpackages it may import ("" = the sdv_bmt root module)
LAYER_MATRIX: dict[str, set[str]] = {
    "": set(),
    "contracts": {"contracts"},
    "input": {"contracts", "input"},
    "workload": {"contracts", "workload"},
    "dut": {"contracts", "dut"},
    "stress": {"contracts", "stress"},
    "observation": {"contracts", "observation"},
    "evidence": {"", "contracts", "observation", "evidence"},
    "metrics": {"", "contracts", "observation", "evidence", "metrics"},
    "runner": {"", "contracts", "observation", "evidence", "runner"},
    "cli": {"", "contracts", "input", "workload", "dut", "stress", "observation",
            "evidence", "metrics", "runner", "cli"},
}

# Standard-library modules that would leak realtime/async/process/telemetry or
# nondeterminism into the Stage 1A core.
FORBIDDEN_STDLIB = {
    "threading", "_thread", "asyncio", "concurrent", "multiprocessing", "subprocess",
    "signal", "socket", "selectors", "sched", "time", "random", "secrets",
    "importlib", "pkgutil", "ctypes", "resource", "faulthandler", "mmap",
}

FORBIDDEN_THIRD_PARTY_PREFIXES = (
    "numpy", "torch", "onnxruntime", "onnx", "tensorflow", "scipy", "pandas",
    "mlperf", "mlperf_loadgen", "carla", "rclpy", "ros2", "autoware", "furiosa",
)

YAML_ALLOWED_MODULES = {"sdv_bmt.runner.plan"}

# Stage 1R-G: module -> forbidden stdlib modules it may import
STAGE_1RG_STDLIB_EXCEPTIONS = {
    "sdv_bmt.runner.realtime": {"time"},
    "sdv_bmt.dut.native": {"time", "subprocess"},
}


@dataclass(frozen=True)
class ImportRef:
    module: str  # importing module
    target: str  # absolute imported module name
    lineno: int


def _module_name(path: Path) -> tuple[str, bool]:
    rel = path.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _source_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _known_modules() -> set[str]:
    return {_module_name(p)[0] for p in _source_files()}


def collect_imports() -> list[ImportRef]:
    refs: list[ImportRef] = []
    for path in _source_files():
        module, is_pkg = _module_name(path)
        package = module if is_pkg else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs.append(ImportRef(module, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_parts = package.split(".")
                    if node.level > 1:
                        base_parts = base_parts[: -(node.level - 1)]
                    base = ".".join(base_parts)
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                refs.append(ImportRef(module, target, node.lineno))
                if node.module is None:
                    # ``from . import x``: x may be a submodule or just an attribute.
                    for alias in node.names:
                        if f"{target}.{alias.name}" in _known_modules():
                            refs.append(ImportRef(module, f"{target}.{alias.name}", node.lineno))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("__import__", "exec", "eval"):
                    refs.append(ImportRef(module, f"<dynamic:{node.func.id}>", node.lineno))
    return refs


def _subpackage(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _violations() -> list[str]:
    out: list[str] = []
    for ref in collect_imports():
        top = ref.target.split(".")[0]
        where = f"{ref.module}:{ref.lineno} imports {ref.target}"
        if ref.target.startswith("<dynamic:"):
            out.append(f"{where} (dynamic import/eval is forbidden)")
        elif top == PACKAGE:
            src = _subpackage(ref.module)
            dst = _subpackage(ref.target)
            if src not in LAYER_MATRIX:
                out.append(f"{where} (module outside the layer matrix)")
            elif dst not in LAYER_MATRIX[src]:
                out.append(f"{where} (layer '{src or 'root'}' may not import '{dst or 'root'}')")
        elif top in sys.stdlib_module_names:
            if top in FORBIDDEN_STDLIB and top not in STAGE_1RG_STDLIB_EXCEPTIONS.get(ref.module, set()):
                out.append(f"{where} (forbidden stdlib module in the core)")
        elif top == "yaml":
            if ref.module not in YAML_ALLOWED_MODULES:
                out.append(f"{where} (PyYAML allowed only in runner/plan.py)")
        else:
            out.append(f"{where} (third-party import outside the Stage 1A allowlist)")
    return out


def test_import_graph_respects_layer_matrix_and_allowlist():
    assert _violations() == []


def test_contracts_import_only_stdlib_and_contracts():
    for ref in collect_imports():
        if _subpackage(ref.module) == "contracts":
            top = ref.target.split(".")[0]
            assert top in sys.stdlib_module_names or ref.target.startswith("sdv_bmt.contracts"), ref


def test_no_forbidden_third_party_anywhere():
    for ref in collect_imports():
        assert not ref.target.startswith(FORBIDDEN_THIRD_PARTY_PREFIXES), ref


def test_metrics_does_not_import_runner_or_dut():
    for ref in collect_imports():
        if _subpackage(ref.module) == "metrics":
            assert _subpackage(ref.target) not in {"runner", "dut", "input", "stress", "workload"}, ref


def test_component_packages_do_not_import_each_other():
    components = {"input", "stress", "dut", "workload"}
    for ref in collect_imports():
        src, dst = _subpackage(ref.module), _subpackage(ref.target)
        if src in components and ref.target.startswith(PACKAGE):
            assert dst in {src, "contracts"}, ref


# host clock reads are allowed only where Stage 1R-G needs them; sleeping is never allowed
HOST_CLOCK_FILES = {SRC_ROOT / "runner" / "realtime.py", SRC_ROOT / "dut" / "native.py"}


def test_no_time_sleep_or_wall_clock_timing_calls():
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        assert "time.sleep" not in text and "process_time" not in text, f"{path}: sleeps or uses process time"
        if path not in HOST_CLOCK_FILES:
            for needle in ("perf_counter", "time.monotonic", "time.time", "clock_gettime"):
                assert needle not in text, f"{path}: contains {needle}"


def test_matrix_self_check_detects_a_violation(tmp_path, monkeypatch):
    # Guard against a vacuous checker: a synthetic forbidden import must be reported.
    fake_root = tmp_path / "sdv_bmt"
    (fake_root / "metrics").mkdir(parents=True)
    (fake_root / "__init__.py").write_text("")
    (fake_root / "metrics" / "__init__.py").write_text("")
    (fake_root / "metrics" / "bad.py").write_text(
        "import threading\nfrom ..runner import core\nimport numpy\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "SRC_ROOT", fake_root)
    found = _violations()
    assert any("threading" in v for v in found)
    assert any("may not import 'runner'" in v for v in found)
    assert any("numpy" in v for v in found)


NATIVE_ROOT = SRC_ROOT.parents[1] / "native"
VENDOR_HEADERS = ("cuda.h", "cuda_runtime", "nvml.h", "nvrtc.h", "nvtx3", "cupti")


def test_vendor_headers_only_in_the_native_nvidia_backend():
    # portable semantics, NVIDIA-specific observation: CUDA/NVML/NVRTC/NVTX/CUPTI stay in native/src/nvidia
    offenders = []
    for path in sorted(NATIVE_ROOT.rglob("*")):
        if path.suffix not in (".hpp", ".cpp") or "build" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#include") and any(h in line for h in VENDOR_HEADERS):
                if "nvidia" not in path.relative_to(NATIVE_ROOT).parts and path.name != "main_nvidia.cpp":
                    offenders.append(f"{path.relative_to(NATIVE_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_no_ctypes_or_vendor_python_bindings():
    for ref in collect_imports():
        assert ref.target.split(".")[0] != "ctypes", ref
