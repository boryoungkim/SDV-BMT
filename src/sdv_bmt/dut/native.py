"""Client of the native measurement plane (approved architecture: C++ measurement plane, Python analysis plane).

A native runner executable (``native/``, C++) executes DUT.prepare, input staging, preconditioning and
MEASURE/DRAIN in its own process.  This module starts it with a resolved run specification, sends one
JSON command per lifecycle phase on stdin and reads exactly one JSON reply line per command on stdout.
No Python code runs per request: during MEASURE/DRAIN the Python control process only waits for the
``measure`` reply.  The native runner appends the raw events of its window to ``events.jsonl`` and the
outputs to ``outputs.bin`` while the Python recorder is suspended (single writer at any time).

A reply that does not arrive within the phase's time limit kills the native process (a watchdog the
in-process design did not have); the driver releases device resources at process exit.

This module is backend-neutral: the NVIDIA and the fake (mock) adapters choose the executable and add
their backend options to the specification.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ..contracts.ids import canonical_json_bytes, digest_of, sha256_hex
from ..contracts.protocols import LifecycleError
from ..contracts.records import ErrorInfo, InputItem, PreparedInfo, RealtimeRunContext, WindowHandover, WindowResult
from ..contracts.specs import DUTConfig, WorkloadSpec, workload_identity_digest

NATIVE_SPEC_SCHEMA = "sdv-bmt-native-run/1"
REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_DIR = REPO_ROOT / "native"


class NativeRunnerError(RuntimeError):
    """The native runner could not be started, did not reply in time, or replied with an error."""


def native_binary(name: str) -> Path:
    """The native runner executable ``name`` (``$SDV_BMT_NATIVE_DIR`` or ``native/build``)."""
    return Path(os.environ.get("SDV_BMT_NATIVE_DIR", NATIVE_DIR / "build")) / name


def native_source_tree_sha256() -> str:
    """SHA-256 over the native sources (``native/src``, ``native/Makefile``) by relative path."""
    files = sorted(p for p in NATIVE_DIR.rglob("*") if p.is_file() and "build" not in p.relative_to(NATIVE_DIR).parts
                   and p.suffix in (".hpp", ".cpp", ""))
    listing = [{"path": str(p.relative_to(NATIVE_DIR)), "sha256": sha256_hex(p.read_bytes())} for p in files]
    return digest_of(listing)


def native_identity(binary: Path) -> dict[str, Any]:
    """Identity of the native measurement plane: executable and source tree digests."""
    if not binary.is_file():
        raise NativeRunnerError(f"native runner {binary} is not built (run: make -C native)")
    return {"native.runner": binary.name, "native.runner_sha256": sha256_hex(binary.read_bytes()),
            "native.source_tree_sha256": native_source_tree_sha256()}


def describe_native(binary: Path, args: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    """Run ``<binary> describe <args>`` (no device context is created) and return its JSON line."""
    try:
        proc = subprocess.run([str(binary), "describe", json.dumps(args)], capture_output=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NativeRunnerError(f"{binary.name} describe failed: {exc}") from None
    try:
        reply = json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise NativeRunnerError(f"{binary.name} describe gave no JSON reply: {proc.stderr.decode(errors='replace')}") from None
    if proc.returncode != 0 or reply.get("ok") is False:
        raise NativeRunnerError(f"{binary.name} describe: {reply.get('error', reply)}")
    return reply


@dataclass(frozen=True)
class NativePreparedInput:
    """A prepared logical input written to the staging file (staged into device-visible memory natively)."""

    item_index: int
    offset: int
    nbytes: int
    sha256: str


def _error(reply: dict[str, Any]) -> ErrorInfo:
    err = reply.get("error") or {}
    return ErrorInfo(str(err.get("code", "native_error")), str(err.get("message", reply)))


class NativeProcess:
    """One native runner process speaking the phase protocol (one JSON line per command and reply)."""

    def __init__(self, binary: Path, spec_path: Path, start_timeout_s: float = 60.0) -> None:
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen([str(binary), "run", str(spec_path)], stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=self._stderr)
        except OSError as exc:
            raise NativeRunnerError(f"cannot start {binary}: {exc}") from None
        self._buffer = b""
        self.ready = self._read("ready", start_timeout_s)
        if not self.ready.get("ok"):
            self.kill()
            raise NativeRunnerError(f"native runner did not start: {_error(self.ready).message}")

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def stderr_text(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace")[-4000:]

    def _read(self, expect: str, timeout_s: float) -> dict[str, Any]:
        fd = self._proc.stdout.fileno()  # type: ignore[union-attr]
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                raise NativeRunnerError(f"no '{expect}' reply within {timeout_s:.0f} s; native runner killed")
            readable, _, _ = select.select([fd], [], [], remaining)
            if readable:
                chunk = os.read(fd, 65536)
                if not chunk:
                    code = self._proc.wait()
                    raise NativeRunnerError(f"native runner exited ({code}) before replying '{expect}': "
                                            f"{self.stderr_text()}")
                self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        reply = json.loads(line.decode("utf-8"))
        if reply.get("reply") not in (expect, "startup"):
            raise NativeRunnerError(f"expected reply '{expect}', got {reply.get('reply')!r}")
        return reply

    def command(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if not self.alive:
            raise NativeRunnerError(f"native runner is not running: {self.stderr_text()}")
        try:
            self._proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except OSError as exc:
            raise NativeRunnerError(f"cannot send '{payload.get('command')}': {exc}") from None
        return self._read(str(payload["command"]), timeout_s)

    def wait_exit(self, timeout_s: float = 30.0) -> int:
        try:
            return self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.kill()
            return self._proc.wait()

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


def _staging_dir() -> str:
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()


class NativeRealtimeAdapter:
    """Base of the native DUT adapters: lifecycle over the phase protocol.

    Subclasses set ``ADAPTER``, ``VERSION``, ``BINARY`` and implement ``describe``, ``capabilities``,
    ``measurement_declaration``, ``backend_spec`` and ``realization_evidence``.
    """

    ADAPTER = ""
    VERSION = ""
    BINARY = ""
    START_TIMEOUT_S = 60.0
    PREPARE_TIMEOUT_S = 300.0
    STAGE_TIMEOUT_S = 600.0
    CLEANUP_TIMEOUT_S = 60.0

    def __init__(self, config: DUTConfig) -> None:
        if config.adapter != self.ADAPTER:
            raise ValueError(f"DUT adapter must be {self.ADAPTER!r}")
        self._config = config
        self._context: Optional[RealtimeRunContext] = None
        self._process: Optional[NativeProcess] = None
        self._spec_path: Optional[Path] = None
        self._staging: Optional[Any] = None
        self._staging_path: Optional[str] = None
        self._staged: list[NativePreparedInput] = []
        self._state = "created"
        self.prepare_facts: dict[str, Any] = {}
        self.realized_affinity: tuple[int, ...] = ()
        self.spec_digest: Optional[str] = None

    # -------------------------------------------------------------- subclass hooks
    def backend_spec(self, workload: WorkloadSpec) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    # -------------------------------------------------------------- lifecycle
    def bind_run(self, context: RealtimeRunContext) -> None:
        if self._state != "created":
            raise LifecycleError("bind_run() must precede prepare()")
        self._context = context

    def _spec(self, workload: WorkloadSpec) -> dict[str, Any]:
        ctx = self._context
        assert ctx is not None
        return {
            "schema": NATIVE_SPEC_SCHEMA,
            "anchor_monotonic_ns": ctx.anchor_monotonic_ns,
            "cpu_affinity": list(ctx.cpu_affinity),
            "schedule": ctx.schedule,
            "run": ctx.run,
            "precondition_requests": ctx.precondition_requests,
            "files": {"events": ctx.events_path, "outputs": ctx.outputs_path},
            "output_pipeline": ctx.output_pipeline,
            "backend": {"name": self.ADAPTER, "options": dict(self._config.runtime_options)},
            **self.backend_spec(workload),
        }

    def prepare(self, workload: WorkloadSpec, config: DUTConfig) -> PreparedInfo:
        if self._state != "created":
            raise LifecycleError(f"prepare() not allowed in state {self._state!r}")
        if config != self._config:
            raise ValueError("prepare() config differs from the adapter's construction config")
        if self._context is None:
            raise LifecycleError("prepare() requires bind_run() (the native run context) first")
        spec = self._spec(workload)
        # deterministic part of the specification (no anchor, no paths) identifies what was executed
        self.spec_digest = digest_of({k: v for k, v in spec.items() if k not in ("anchor_monotonic_ns", "files")})
        fd, path = tempfile.mkstemp(prefix="sdv-bmt-native-spec-", suffix=".json", dir=_staging_dir())
        with os.fdopen(fd, "wb") as fh:
            fh.write(canonical_json_bytes(spec))
        self._spec_path = Path(path)
        self._state = "starting"
        self._process = NativeProcess(native_binary(self.BINARY), self._spec_path, self.START_TIMEOUT_S)
        self.realized_affinity = tuple(self._process.ready.get("realized_affinity", ()))
        reply = self._process.command({"command": "prepare"}, self.PREPARE_TIMEOUT_S)
        if not reply.get("ok"):
            raise NativeRunnerError(f"native DUT.prepare failed: {_error(reply).code}: {_error(reply).message}")
        self.prepare_facts = reply.get("facts", {})
        fd, self._staging_path = tempfile.mkstemp(prefix="sdv-bmt-staging-", suffix=".bin", dir=_staging_dir())
        self._staging = os.fdopen(fd, "wb")
        self._state = "prepared"
        return PreparedInfo(workload_identity_digest=workload_identity_digest(workload),
                            notes=(f"native measurement plane {self.BINARY}; spec digest {self.spec_digest}",))

    def prepare_input(self, task_input: object, item: InputItem) -> NativePreparedInput:
        if self._state != "prepared" or self._staging is None:
            raise LifecycleError("prepare_input() requires prepare() first")
        if not isinstance(task_input, (bytes, bytearray)):
            raise TypeError("the native measurement plane stages bytes task inputs")
        data = bytes(task_input)
        offset = self._staging.tell()
        self._staging.write(data)
        prepared = NativePreparedInput(item.item_index, offset, len(data), hashlib.sha256(data).hexdigest())
        self._staged.append(prepared)
        return prepared

    def commit_inputs(self, plan: Sequence[tuple[int, Optional[str], Optional[NativePreparedInput]]],
                      output_contexts: Optional[dict[int, Any]] = None) -> dict[int, ErrorInfo]:
        """Stage every planned item natively (pinned or device memory, SHA-256 re-verified).

        ``plan`` lists (item_index, failure_stage, prepared) in planned order; ``output_contexts`` maps an
        item index to the per-item facts the workload's native output pipeline needs (e.g. the original
        image size).  Returns the items whose native staging failed (their stage is ``prepare_input``)."""
        if self._state != "prepared" or self._process is None or self._staging is None:
            raise LifecycleError("commit_inputs() requires prepare() first")
        self._staging.close()
        items = []
        for index, failure, prepared in plan:
            if failure is not None or prepared is None:
                row: dict[str, Any] = {"item_index": index, "failure_stage": failure or "prepare_input"}
            else:
                row = {"item_index": index, "offset": prepared.offset, "nbytes": prepared.nbytes,
                       "sha256": prepared.sha256}
            if output_contexts is not None and index in output_contexts:
                row["output_context"] = output_contexts[index]
            items.append(row)
        try:
            reply = self._process.command({"command": "stage", "path": self._staging_path, "items": items},
                                          self.STAGE_TIMEOUT_S)
        finally:
            self._remove_staging()
        if not reply.get("ok"):
            raise NativeRunnerError(f"native staging failed: {_error(reply).message}")
        failed = {}
        for row in reply["items"]:
            if not row["ok"] and row.get("stage") == "prepare_input":
                failed[row["item_index"]] = _error(row)
        self._state = "staged"
        return failed

    def run_window(self, handover: WindowHandover) -> WindowResult:
        """Precondition, backend snapshot, MEASURE/DRAIN and snapshot in the native process."""
        if self._state != "staged" or self._process is None or self._context is None:
            raise LifecycleError("run_window() requires staged inputs")
        ctx = self._context
        offsets = ctx.schedule.get("offsets_ns") or [0]
        span_s = (max(offsets) + ctx.schedule["lead_ns"]) / 1e9
        n = max(1, len(offsets))
        limit = (span_s + n * ctx.run["request_timeout_ns"] / 1e9 * (1 if ctx.schedule["mode"] == "asap" else 0)
                 + ctx.run["drain_ns"] / 1e9 + ctx.precondition_requests * ctx.run["request_timeout_ns"] / 1e9 + 120.0)
        reply = self._process.command({"command": "measure", "next_seq": handover.next_seq,
                                       "last_t_run_ns": handover.last_t_run_ns,
                                       "output_offset": handover.output_offset}, limit)
        self._state = "measured"
        return WindowResult(
            ok=bool(reply.get("ok")), failed_step=reply.get("failed_step"),
            error=_error(reply) if not reply.get("ok") else None,
            next_seq=int(reply["next_seq"]), last_t_run_ns=int(reply["last_t_run_ns"]),
            output_offset=int(reply["output_offset"]), summary=reply.get("summary"),
            native_counts=reply.get("native_counts"), precondition=reply.get("precondition"))

    def cleanup(self) -> None:
        if self._state == "cleaned":
            return
        self._remove_staging()
        process, self._process = self._process, None
        self._state = "cleaned"
        try:
            if process is not None and process.alive:
                reply = process.command({"command": "cleanup"}, self.CLEANUP_TIMEOUT_S)
                process.wait_exit()
                if not reply.get("ok"):
                    raise NativeRunnerError(f"native DUT.cleanup: {reply.get('facts', reply)}")
            elif process is not None:
                raise NativeRunnerError(f"native runner had exited before cleanup: {process.stderr_text()}")
        finally:
            if process is not None:
                process.kill()
            if self._spec_path is not None:
                self._spec_path.unlink(missing_ok=True)

    def _remove_staging(self) -> None:
        if self._staging is not None and not self._staging.closed:
            self._staging.close()
        if self._staging_path is not None:
            Path(self._staging_path).unlink(missing_ok=True)
            self._staging_path = None

    def input_override(self, contract: object, implementation_id: str) -> None:
        return None  # no adapter preprocessing override is implemented

    def prepared_input_digest(self, prepared_input: object) -> str:
        if not isinstance(prepared_input, NativePreparedInput):
            raise TypeError(f"not a {self.ADAPTER} prepared input")
        return prepared_input.sha256
