"""tools/run_model.sh: the model names, the dry run (configuration files, no writes), and the flow with stub
commands in place of sdv-bmt and the table command (runs, evaluation, table, refusal of existing runs, the timed-run
marker; MLA400: the native runner, device node and compiled model checks, and the refusal while another process
holds the NPU device).  No device is used."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from tests._helpers import REPO_ROOT

SCRIPT = REPO_ROOT / "tools" / "run_model.sh"
MODELS = ["yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x", "deeplabv3plus-r18", "deeplabv3plus-r50",
          "deeplabv3plus-r101", "centerpoint-pillar02", "llama-3.2-1b-instruct", "llama-3.2-3b-instruct",
          "llama-3.1-8b-instruct"]
MLA400 = ["yolox-tiny-mla400", "yolox-s-mla400", "yolox-m-mla400", "yolox-l-mla400", "yolox-x-mla400"]
LLM_ENV = ("SDV_BMT_MODELS", "SDV_BMT_DATASETS", "SDV_BMT_CACHE", "VLLM_PY", "FURIOSA_PY", "HF_HOME")

STUB_BMT = """#!/usr/bin/env bash
case $1 in
  validate) echo '{"status": "valid"}' ;;
  run) exp=$(basename "$2" .yaml); d=$4/$exp/run-$(date +%s%N); mkdir -p "$d"; touch "$d/manifest.json"
       printf '{\\n  "execution_status": "completed",\\n  "measurement_status": "valid"\\n}\\n' ;;
  evaluate) mkdir -p "$2/evaluations/eval-stub" ;;
esac
"""
STUB_PY = """#!/usr/bin/env bash
while [ "$1" != "--out" ]; do shift; done
printf 'row,metric_name\\nheadline,DUT service time (median)\\n' > "$2"
echo "   DUT service time (median) - 1.000 ms" >&2
"""


def _run(args, env=None):
    base = {k: v for k, v in os.environ.items()
            if k not in LLM_ENV + ("R", "REPS", "BMT", "PY", "MARK", "MLA400_DEVICE", "SDV_BMT_NATIVE_DIR")}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env={**base, **(env or {})},
                          timeout=120)


def test_list_names_the_12_models_and_the_5_mla400_models():
    proc = _run(["list"])
    assert proc.returncode == 0 and proc.stdout.split() == MODELS + MLA400
    assert _run(["no-such-model"]).returncode == 2


def test_dry_run_checks_every_configuration_and_writes_nothing(tmp_path):
    out = tmp_path / "bench"
    proc = _run(["all", "--dry-run"], {"R": str(out)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not out.exists()
    assert "Model-level Accelerator Benchmark" in proc.stdout and "\u2500\u2500 [12/12] llama-3.1-8b-instruct " in proc.stdout
    banner = proc.stdout.splitlines()[1:6]  # the box: SDV-BMT in 3 rows of block letters
    assert banner[0].lstrip().startswith("\u250c") and banner[4].lstrip().startswith("\u2514")
    assert len({len(row) for row in banner}) == 1 and "\u2588\u2580\u2580 \u2588\u2580\u2584" in banner[1]
    for name in MODELS[:8]:
        cfg = f"configs/experiments/{name}-baseline.yaml"
        assert (REPO_ROOT / cfg).is_file() and f"validate {cfg}" in proc.stdout
    assert "configs/experiments/centerpoint-pillar02-val-series.yaml" in proc.stdout
    assert proc.stdout.count("--catalog sdv-bmt-model-metric-set-v1.2") == 2
    for name in MODELS[9:]:
        assert f"run_baselines.sh gpu {name}" in proc.stdout and f"run_baselines.sh rngd {name}" in proc.stdout
        for c in REPO_ROOT.glob(f"configs/llm/experiments/{name}.*.yaml"):
            assert c.is_file()
    assert proc.stdout.count("sdv_bmt.cli.table") == 9 and proc.stdout.count("sdv_bmt.llm table") == 3


def test_flow_with_stub_commands_refuses_existing_runs_and_a_held_marker(tmp_path):
    for name, text in (("sdv-bmt", STUB_BMT), ("python", STUB_PY)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    out, mark = tmp_path / "bench", tmp_path / "MARK"
    env = {"R": str(out), "REPS": "2", "BMT": str(tmp_path / "sdv-bmt"), "PY": str(tmp_path / "python"),
           "MARK": str(mark)}
    proc = _run(["yolox-s"], env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = sorted((out / "vision" / "yolox-s-baseline").glob("run-*"))
    assert len(runs) == 2 and all((r / "evaluations").is_dir() for r in runs)
    assert "run 2/2  completed/valid" in proc.stdout and "evaluated 2 (sdv-bmt-model-metric-set-v1.1)" in proc.stdout
    assert (out / "tables" / "yolox-s.metrics.csv").is_file()
    assert "DUT service time (median) - 1.000 ms" in proc.stdout
    assert not mark.exists()  # the marker is released
    again = _run(["yolox-s"], env)
    assert again.returncode == 3 and "already has runs" in again.stdout
    mark.write_text("what=other\n")
    held = _run(["yolox-m"], env)
    assert held.returncode == 3 and "another timed run is in progress" in held.stdout
    assert mark.read_text() == "what=other\n" and not (out / "vision" / "yolox-m-baseline").exists()
    missing = _run(["llama-3.2-1b-instruct"], env)
    assert missing.returncode == 1 and "export" in missing.stdout  # no LLM environment: nothing is started


def test_mla400_dry_run_names_the_plans_the_device_and_the_compiled_models(tmp_path):
    proc = _run(["all-mla400", "--dry-run"], {"R": str(tmp_path / "bench")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all 5 (Mobilint MLA400)" in proc.stdout and not (tmp_path / "bench").exists()
    for name in MLA400:
        cfg = f"configs/experiments/{name}-baseline.yaml"
        assert (REPO_ROOT / cfg).is_file() and f"validate {cfg}" in proc.stdout
        short = name.split("-")[1]
        assert f"{name}: NPU /dev/aries2, compiled model " in proc.stdout and f"yolox_{short}_head9.mxq" in proc.stdout
    assert proc.stdout.count("--catalog sdv-bmt-model-metric-set-v1.1") == 5
    assert proc.stdout.count("sdv_bmt.cli.table") == 5 and "llama" not in proc.stdout


def test_mla400_flow_checks_the_host_and_refuses_a_held_device(tmp_path):
    for name, text in (("sdv-bmt", STUB_BMT), ("python", STUB_PY)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    native, home, device = tmp_path / "native", tmp_path / "home", tmp_path / "aries-dev"
    env = {"R": str(tmp_path / "bench"), "REPS": "1", "BMT": str(tmp_path / "sdv-bmt"), "PY": str(tmp_path / "python"),
           "MARK": str(tmp_path / "MARK"), "SDV_BMT_NATIVE_DIR": str(native), "HOME": str(home),
           "MLA400_DEVICE": str(device)}
    missing = _run(["yolox-s-mla400"], env)
    assert missing.returncode == 1 and "build it first: tools/mobilint/build_native.sh" in missing.stdout
    assert "does not exist" in missing.stdout and "yolox_s_head9.mxq missing" in missing.stdout
    assert not (tmp_path / "bench" / "vision" / "yolox-s-mla400-baseline").exists()  # nothing started
    native.mkdir()
    (native / "sdv-bmt-native-mobilint").write_text("#!/bin/sh\n")
    (native / "sdv-bmt-native-mobilint").chmod(0o755)
    for short in ("s", "m"):
        mxq = home / "sdv_assets" / "compile" / "head9" / f"yolox_{short}_head9.mxq"
        mxq.parent.mkdir(parents=True, exist_ok=True)
        mxq.write_bytes(b"stub")
    device.write_text("")
    ok = _run(["yolox-s-mla400"], env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "run 1/1  completed/valid" in ok.stdout and (tmp_path / "bench" / "tables" / "yolox-s-mla400.metrics.csv").is_file()
    assert "Mobilint SDK license: do not publish these results" in ok.stdout
    holder = subprocess.Popen(["bash", "-c", f"exec 3<'{device}'; sleep 60"])
    try:
        for _ in range(50):  # until the holder has the device open
            if any(os.path.realpath(f"/proc/{holder.pid}/fd/{fd}") == str(device)
                   for fd in os.listdir(f"/proc/{holder.pid}/fd")):
                break
            time.sleep(0.1)
        held = _run(["yolox-m-mla400"], env)
    finally:
        holder.kill()
        holder.wait()
    assert held.returncode == 3 and f"{device} is held by: {holder.pid} " in held.stdout
    assert not list((tmp_path / "bench" / "vision").glob("yolox-m-mla400-baseline/run-*"))  # no run started
