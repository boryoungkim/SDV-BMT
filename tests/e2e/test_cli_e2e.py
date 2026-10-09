"""E2E: validate -> virtual run -> bundle -> evaluate through the CLI, each step in
its own process (S1A-1 determinism across processes, S1A-11 MOCK propagation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sdv_bmt.contracts.records import RAW_EVIDENCE_FILES
from sdv_bmt.evidence.manifest import canonical_manifest_content
from tests._helpers import LEGACY_CATALOG, REPO_ROOT, plan_path

SCRIPT = Path(sys.executable).parent / "sdv-bmt"


def cli(*args: str, cwd: Path = REPO_ROOT) -> tuple[int, dict, str]:
    proc = subprocess.run([sys.executable, "-m", "sdv_bmt.cli.main", *args], cwd=cwd,
                          capture_output=True, text=True)
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, payload, proc.stderr


def run_one(plan: str, out: Path) -> Path:
    code, payload, _ = cli("run", str(plan_path(plan)), "--out", str(out))
    assert code == 0, payload
    assert payload["status"] == "executed" and len(payload["runs"]) == 1
    return Path(payload["runs"][0]["run_dir"])


def test_console_script_is_installed():
    assert SCRIPT.is_file()
    proc = subprocess.run([str(SCRIPT), "--version"], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "sdv-bmt 0.1.0"


def test_help_lists_only_validate_run_evaluate():
    proc = subprocess.run([sys.executable, "-m", "sdv_bmt.cli.main", "--help"], capture_output=True, text=True)
    assert "{validate,run,evaluate}" in proc.stdout and "analyze" not in proc.stdout


def test_validate_reports_mock_and_writes_nothing(tmp_path):
    code, payload, stderr = cli("validate", str(plan_path("g2-recorded-scenario-overload")), cwd=tmp_path)
    assert code == 0 and payload["status"] == "valid"
    assert payload["execution_kind"] == "mock" and "MOCK" in payload["notice"] and "[MOCK]" in stderr
    assert payload["schedule"]["planned_offsets_ns"] == [0, 33_000_000, 66_000_000, 99_000_000]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "plan,message",
    [
        ("reject-dataset-recorded", "recorded replay is not allowed for dataset input"),
        ("reject-timestamp-regression", "source timestamp regression"),
    ],
)
def test_rejections_exit_2_and_create_nothing(tmp_path, plan, message):
    code, payload, _ = cli("validate", str(plan_path(plan)))
    assert code == 2 and payload["status"] == "rejected" and message in payload["issues"][0]
    code, payload, _ = cli("run", str(plan_path(plan)), "--out", str(tmp_path / "runs"))
    assert code == 2 and payload["execution_kind"] == "mock"
    assert not (tmp_path / "runs").exists()


def test_validate_run_evaluate_pipeline_is_deterministic_across_processes(tmp_path):
    first = run_one("g2-recorded-scenario-overload", tmp_path / "a")
    second = run_one("g2-recorded-scenario-overload", tmp_path / "b")
    assert first.name != second.name  # volatile run IDs differ
    for name in RAW_EVIDENCE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    ma = json.loads((first / "manifest.json").read_text())
    mb = json.loads((second / "manifest.json").read_text())
    assert canonical_manifest_content(ma) == canonical_manifest_content(mb)
    digests = []
    for run_dir in (first, second):
        code, payload, _ = cli("evaluate", str(run_dir), "--catalog", LEGACY_CATALOG)
        assert code == 0 and payload["execution_kind"] == "mock"
        digests.append(payload["metrics_content_sha256"])
        assert payload["metrics"]["bmt.rate.drop"]["value"] == 0.5
    assert digests[0] == digests[1]


def test_bundle_tree_is_complete(tmp_path):
    run_dir = run_one("g1-dataset-fixed-rate-nominal", tmp_path)
    code, payload, _ = cli("evaluate", str(run_dir), "--catalog", LEGACY_CATALOG)
    assert code == 0
    exp_dir = run_dir.parent
    assert (exp_dir / "experiment.json").is_file()
    expected = set(RAW_EVIDENCE_FILES) | {"manifest.json", "evaluations"}
    assert {p.name for p in run_dir.iterdir()} == expected
    eval_dir = run_dir / "evaluations" / payload["eval_id"]
    assert {p.name for p in eval_dir.iterdir()} == {"metrics.json", "eval_manifest.json"}
    assert payload["metrics"]["bmt.output.identity_rate"] == {"status": "valid", "unit": "ratio", "value": 1.0}


def test_failed_stress_run_exits_1_and_evaluates_to_not_collected(tmp_path):
    code, payload, _ = cli("run", str(plan_path("g4-stress-verify-failure")), "--out", str(tmp_path))
    assert code == 1
    run = payload["runs"][0]
    assert (run["execution_status"], run["execution_reason"]) == ("failed", "stress_verify_failed")
    code, payload, _ = cli("evaluate", run["run_dir"])
    assert code == 0
    assert {m["status"] for m in payload["metrics"].values()} == {"not_collected"}


def test_evaluate_with_deadline_override(tmp_path):
    run_dir = run_one("g2-recorded-scenario-overload", tmp_path)
    override = tmp_path / "deadline-120ms.yaml"
    override.write_text("schema_version: 1\ndeadline: {policy: frame_period, period_ns: 120_000_000}\n")
    code, payload, _ = cli("evaluate", str(run_dir), "--eval-config", str(override), "--catalog", LEGACY_CATALOG)
    assert code == 0 and payload["metrics"]["bmt.deadline.miss_rate"]["value"] == 0.5
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\ndeadline: null\noutput_reference: {kind: none}\n")
    code, payload, _ = cli("evaluate", str(run_dir), "--eval-config", str(bad))
    assert code == 2 and "unknown field" in payload["issues"][0]


def test_evaluate_refuses_tampered_bundle(tmp_path):
    run_dir = run_one("g3-timeout-late-completion", tmp_path)
    with open(run_dir / "outputs.bin", "ab") as fh:
        fh.write(b"x")
    code, payload, _ = cli("evaluate", str(run_dir))
    assert code == 2 and "raw integrity violated" in payload["issues"][0]
    assert not (run_dir / "evaluations").exists()


def test_restore_failure_blocks_second_invocation(tmp_path):
    configs = tmp_path / "configs"
    import shutil

    shutil.copytree(REPO_ROOT / "configs", configs)
    stress = configs / "stress" / "fake-virtual-axis.yaml"
    stress.write_text(stress.read_text().replace("provider_options: {}", "provider_options: {fail_phase: restore}"))
    plan = configs / "experiments" / "scenario-two-episodes-fake-stress.yaml"
    code, payload, _ = cli("run", str(plan), "--out", str(tmp_path / "runs"))
    assert code == 1 and payload["runs"][0]["execution_reason"] == "stress_restore_failed"
    code, payload, _ = cli("run", str(plan), "--out", str(tmp_path / "runs"))
    assert code == 2 and "no successful stress restore" in payload["rejected"][0]


# ---------------------------------------------------------------- explicit input pipeline


def test_override_unavailable_is_rejected_by_validate_and_run(tmp_path):
    code, payload, _ = cli("validate", str(plan_path("reject-override-unavailable")))
    assert code == 2 and "provides no input override" in payload["issues"][0]
    code, payload, _ = cli("run", str(plan_path("reject-override-unavailable")), "--out", str(tmp_path / "runs"))
    assert code == 2 and not (tmp_path / "runs").exists()


def test_override_pipeline_end_to_end(tmp_path):
    code, payload, _ = cli("validate", str(plan_path("g1-adapter-override")))
    assert code == 0
    assert payload["input_pipeline"]["effective"]["owner"] == "dut_adapter"
    assert payload["input_pipeline"]["override"]["implementation_id"] == "fake-fused-prefix-preprocess"
    run_dir = run_one("g1-adapter-override", tmp_path)
    code, payload, _ = cli("evaluate", str(run_dir))
    assert code == 0 and payload["metrics"]["bmt.output.identity_rate"]["value"] == 1.0


def test_override_equivalence_failure_exits_1(tmp_path):
    code, payload, _ = cli("run", str(plan_path("override-equivalence-failure")), "--out", str(tmp_path))
    assert code == 1
    assert payload["runs"][0]["execution_reason"] == "input_equivalence_failed"
