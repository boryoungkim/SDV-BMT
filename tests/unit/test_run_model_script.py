"""tools/run_model.sh: the model names, the dry run (configuration files, no writes), and the flow with stub
commands in place of sdv-bmt and the table command (runs, evaluation, table, refusal of existing runs, the timed-run
marker; Llama: one device per model name, llama-<m> on the GPU and llama-<m>-rngd on the RNGD card, with the refusal
of existing runs per device; MLA400: the native runner, device node and compiled model checks, and the refusal while
another process holds the NPU device).  No device is used."""

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
CP_MINI = "centerpoint-pillar02-mini"
RNGD = ["llama-3.2-1b-instruct-rngd", "llama-3.2-3b-instruct-rngd", "llama-3.1-8b-instruct-rngd"]
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
STUB_LLM = """#!/usr/bin/env bash
# tools/llm/run_baselines.sh DEVICE MODEL with OUT and REPS: REPS speed runs and one MMLU run of that device
case $1 in gpu) s=gpu0-vllm ;; rngd) s=rngd-furiosa ;; esac
log=$OUT/baseline-$1-$(date +%s%N).log
for i in $(seq 1 "$REPS"); do mkdir -p "$OUT/$2.$s.speed/run-$i"; echo '{"run_status": "completed"}' >> "$log"; done
mkdir -p "$OUT/$2.$s.mmlu/run-1"; echo '{"run_status": "completed"}' >> "$log"
"""
STUB_LLM_TABLE = """#!/usr/bin/env bash
args="$*"; while [ "$1" != "--out" ]; do shift; done
printf 'row,metric_name\\nheadline,TTFT\\n# %s\\n' "$args" > "$2"
echo "   Time to first token - 1.000 ms" >&2
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


def _datasets(stdout):
    """The dataset lines of the start header: the values from the "dataset(s)" line to the next labelled line."""
    lines = stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(("  dataset ", "  datasets ")))
    values = [lines[start][19:]]
    for line in lines[start + 1:]:
        if not line.startswith(" " * 19):
            break
        values.append(line[19:])
    return values


def test_list_names_the_12_gpu_models_the_3_rngd_models_and_the_5_mla400_models():
    proc = _run(["list"])
    assert proc.returncode == 0 and proc.stdout.split() == MODELS + [CP_MINI] + RNGD + MLA400
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
    for name in MODELS[9:]:  # all = the 12 models on the GPU: the Llama models with vLLM only
        assert f"run_baselines.sh gpu {name}" in proc.stdout and f"run_baselines.sh rngd {name}" not in proc.stdout
        for c in REPO_ROOT.glob(f"configs/llm/experiments/{name}.gpu0-vllm.*.yaml"):
            assert c.is_file()
    assert "all 12 (GPU)" in proc.stdout and "rngd-furiosa" not in proc.stdout
    assert proc.stdout.count("sdv_bmt.cli.table") == 9 and proc.stdout.count("sdv_bmt.llm table") == 3
    assert _datasets(proc.stdout) == [  # under the models line, each once, as the result table writes them
        "coco-val2017-first1000-416 (1,000 items)", "coco-val2017-first1000 (1,000 items)",
        "cityscapes-val500 (500 items)", "nuscenes-mini-val4 (162 items)", "nuscenes-val (6,019 items)",
        "speed-s1-s5 (synthetic token IDs)", "mmlu-1000 (1,000 items)"]


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
    assert "  models           yolox-s\n  dataset          coco-val2017-first1000 (1,000 items)\n" in proc.stdout
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
    assert _datasets(proc.stdout) == ["coco-val2017-first1000-416 (1,000 items)", "coco-val2017-first1000 (1,000 items)"]


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


def test_rngd_dry_run_runs_the_rngd_card_only(tmp_path):
    for name in RNGD:
        base = name.removesuffix("-rngd")
        proc = _run([name, "--dry-run"], {"R": str(tmp_path / "bench")})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert f"run_baselines.sh rngd {base}" in proc.stdout and "run_baselines.sh gpu" not in proc.stdout
        assert f"llm/{base}.rngd-furiosa.speed" in proc.stdout and f"llm/{base}.rngd-furiosa.mmlu" in proc.stdout
        assert "gpu0-vllm" not in proc.stdout and f"tables/{name}.metrics.csv" in proc.stdout
        for kind in ("speed", "mmlu"):
            assert (REPO_ROOT / f"configs/llm/experiments/{base}.rngd-furiosa.{kind}.yaml").is_file()
        assert _datasets(proc.stdout) == ["speed-s1-s5 (synthetic token IDs)", "mmlu-1000 (1,000 items)"]
    assert not (tmp_path / "bench").exists()


def test_llama_flow_one_device_per_name_and_existing_runs_refused_per_device(tmp_path):
    for name, text in (("run_baselines.sh", STUB_LLM), ("python", STUB_LLM_TABLE)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    out = tmp_path / "bench"
    env = {"R": str(out), "REPS": "2", "PY": str(tmp_path / "python"), "MARK": str(tmp_path / "MARK"),
           "LLM_BASELINES": str(tmp_path / "run_baselines.sh"), **{k: str(tmp_path / k.lower()) for k in LLM_ENV}}
    rngd = _run(["llama-3.2-1b-instruct-rngd"], env)
    assert rngd.returncode == 0, rngd.stdout + rngd.stderr
    assert "llama-3.2-1b-instruct-rngd: rngd (speed 2 runs + MMLU 1 run)" in rngd.stdout
    assert "llama-3.2-1b-instruct-rngd: 3/3 runs completed" in rngd.stdout
    assert len(list((out / "llm").glob("llama-3.2-1b-instruct.rngd-furiosa.speed/run-*"))) == 2
    assert not list((out / "llm").glob("*.gpu0-vllm.*"))
    table = (out / "tables" / "llama-3.2-1b-instruct-rngd.metrics.csv").read_text()
    assert "rngd-furiosa.speed" in table and "rngd-furiosa.mmlu" in table and "gpu0-vllm" not in table
    gpu = _run(["llama-3.2-1b-instruct"], env)  # the RNGD runs of the model in the same R do not block the GPU
    assert gpu.returncode == 0, gpu.stdout + gpu.stderr
    assert "llama-3.2-1b-instruct: gpu (speed 2 runs + MMLU 1 run)" in gpu.stdout
    table = (out / "tables" / "llama-3.2-1b-instruct.metrics.csv").read_text()
    assert "gpu0-vllm.speed" in table and "gpu0-vllm.mmlu" in table and "rngd-furiosa" not in table
    again = _run(["llama-3.2-1b-instruct-rngd"], env)
    assert again.returncode == 3 and "already has runs of llama-3.2-1b-instruct.rngd-furiosa" in again.stdout


def test_centerpoint_mini_runs_the_mini_set_only(tmp_path):
    proc = _run([CP_MINI, "--dry-run"], {"R": str(tmp_path / "bench")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "validate configs/experiments/centerpoint-pillar02-mini-baseline.yaml" in proc.stdout
    assert "val-series" not in proc.stdout and "--catalog sdv-bmt-model-metric-set-v1.2" in proc.stdout
    assert proc.stdout.count("sdv_bmt.cli.table") == 1 and f"tables/{CP_MINI}.metrics.csv" in proc.stdout
    assert _datasets(proc.stdout) == ["nuscenes-mini-val4 (162 items)"] and not (tmp_path / "bench").exists()
    for name, text in (("sdv-bmt", STUB_BMT), ("python", STUB_PY)):
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    out = tmp_path / "bench"
    env = {"R": str(out), "REPS": "1", "BMT": str(tmp_path / "sdv-bmt"), "PY": str(tmp_path / "python"),
           "MARK": str(tmp_path / "MARK")}
    ran = _run([CP_MINI], env)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert len(list((out / "vision" / "centerpoint-pillar02-mini-baseline").glob("run-*"))) == 1
    assert "evaluated 1 (sdv-bmt-model-metric-set-v1.2)" in ran.stdout
    assert (out / "tables" / f"{CP_MINI}.metrics.csv").is_file() and not (out / "vision" / "centerpoint-pillar02-val-series").exists()

