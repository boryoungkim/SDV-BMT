#!/usr/bin/env python3
"""Run one Eagle-N step in the pinned BOS container.

No device is exposed for build, build-runner, describe, validate and evaluate. ``l1 START:COUNT`` runs the offline
L1 numeric harness (tools/eagle_n/l1_native.py) with the device. ``run`` first requires the native qualification and
a staging memory budget; it uses a private hugepage mount backed by the existing pool. The launcher never reserves
RAM, never resets or reloads drivers and never kills a container that holds the NPU.

Host layout (tools/eagle_n/README.md, host setup): ``/opt/sdv_assets/eagle_n`` points to the Eagle-N root, which
holds ``eagle_n_integration_assets`` (executor, plans, receipts, kernel caches) and the evidence root. The container
sees that root at both paths, and the repository checkout at its own path. ``EAGLE_N_SDK_ROOT`` (or ``--sdk``) is the
BOS SDK build tree.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EAGLE_N_ROOT = Path("/opt/sdv_assets/eagle_n")
ASSETS = EAGLE_N_ROOT / "eagle_n_integration_assets"
DEFAULT_EVIDENCE = EAGLE_N_ROOT / "eagle_n_integration_evidence"
IMAGE = "sha256:bb29c5af20279d1ad95805ebfb2061399d409778b3ba54d87b1f102235d6ec8d"
DEVICE = "/dev/bos/0"
STAGED_ITEM_BYTES = 12582912  # staged NHWC BF16 input, 12 MiB per item


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def npu_idle() -> None:
    if not Path(DEVICE).exists():
        raise ValueError("device node absent")
    # fuser exits 1 when no process uses the device, and so does a refused sudo: require sudo -n first.
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0:
        raise ValueError("sudo -n unavailable (run 'sudo -v' first): the NPU user check cannot run")
    r = subprocess.run(["sudo", "-n", "fuser", DEVICE], capture_output=True, text=True)
    if r.returncode != 1:
        raise ValueError("NPU is busy or competing-user query failed: " + r.stdout + r.stderr)


def precheck_device():
    npu_idle()


def precheck(plan, evidence: Path = DEFAULT_EVIDENCE):
    sys.path.insert(0, str(ROOT / "src"))
    from sdv_bmt.runner.plan import load_plan

    rp = load_plan(plan)
    opt = rp.dut.runtime_options
    p = rp.workload.options["device_program"]
    if rp.dut.adapter != "bos_eagle_n":
        raise ValueError("only bos_eagle_n allowed by this launcher")
    if opt.get("trial_unqualified", False):
        if (
            p["variant"] != "r18"
            or rp.input_spec.selection.count != 1
            or rp.plan.realtime.precondition_requests != 0
            or "trial" not in rp.plan.experiment_id
        ):
            raise ValueError(
                "Unqualified direct trial requires R18, one input, zero precondition requests and trial experiment id"
            )
    else:
        q = Path(p["qualification"])
        if not q.is_file() or not p.get("qualification_sha256"):
            raise ValueError("Native model qualification NOT RUN: complete L1 and qualify_native.py, then configure.py")
        receipt = json.loads(q.read_text())
        expected = [
            ("executor_sha256", sha(opt["executor"])),
            ("executor_source_sha256", opt["executor_source_sha256"]),
            ("plan_sha256", sha(p["plan"])),
            ("weights_sha256", sha(p["weights"])),
        ]
        if not receipt.get("passed") or any(receipt.get(k) != v for k, v in expected):
            raise ValueError("native qualification stale or failed")
        if sha(q) != p["qualification_sha256"]:
            raise ValueError("native qualification stale: the receipt differs from the workload pin")
    count = rp.input_spec.selection.count
    if count is None:
        raise ValueError("bounded staging needs explicit input selection count")
    plan_precision = json.loads(Path(p["plan"]).read_text()).get("precision")
    if plan_precision != "bf16":
        raise ValueError(f"plan precision {plan_precision} not supported (BF16 only)")
    size = STAGED_ITEM_BYTES * count
    need = count * int(p["input_bytes"]) + 2 * 2**30
    free = shutil.disk_usage(evidence).free
    if free < need:
        raise ValueError(f"SDV-BMT staging file needs {need} bytes free on disk (inputs + 2 GiB), only {free}")
    if size > opt["staging_budget_bytes"]:
        raise ValueError(
            f"staged host payload {size} exceeds budget {opt['staging_budget_bytes']}; "
            "resolve residency/budget before NPU access"
        )
    npu_idle()
    return rp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["build", "build-runner", "describe", "validate", "run", "evaluate", "l1"])
    p.add_argument("target", nargs="?")
    sdk = os.environ.get("EAGLE_N_SDK_ROOT")
    p.add_argument("--sdk", type=Path, default=Path(sdk) if sdk else None, help="BOS SDK root (EAGLE_N_SDK_ROOT)")
    p.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--kernel-cache", type=Path)
    p.add_argument("--wall-seconds", type=int, default=1800)
    p.add_argument("--rss-probe-slots", type=int, default=0, help="l1 only: extra staging slots for the RSS probe")
    p.add_argument("--dut", type=Path, help="l1 only: DUT config (default R18)")
    p.add_argument("--workload", type=Path, help="l1 only: workload config (default R18)")
    p.add_argument("--executor-out", type=Path, help="build only: write the executor here (inside the assets root)")
    p.add_argument(
        "--sync-each-op",
        action="store_true",
        help="l1 only (diagnosis): synchronize after every plan op during warm-up and log START/DONE",
    )
    p.add_argument(
        "--preflight", action="store_true", help="l1 only: NO_DISPATCH L1 preflight, no program sent to the cores"
    )
    p.add_argument(
        "--plan-override", type=Path, help="l1 --preflight only: plan file to check instead of the workload plan"
    )
    p.add_argument("--golden-extra", type=int, default=0, help="l1 only: extra identical golden replays, 0..16")
    a = p.parse_args()
    if a.sdk is None:
        p.error("the BOS SDK root is required: set EAGLE_N_SDK_ROOT or pass --sdk")
    if a.rss_probe_slots and (a.mode != "l1" or not 0 < a.rss_probe_slots <= 64):
        p.error("--rss-probe-slots: l1 only, 1..64")
    if a.golden_extra and (a.mode != "l1" or a.preflight or not 0 < a.golden_extra <= 16):
        p.error("--golden-extra: l1 device run only, 1..16")
    if a.mode not in ("build", "build-runner", "describe") and not a.target:
        p.error("target plan/bundle is required")
    if (a.dut or a.workload) and a.mode != "l1":
        p.error("--dut/--workload: l1 only")
    if a.executor_out is not None and (
        a.mode != "build" or not a.executor_out.resolve().is_relative_to(ASSETS.resolve())
    ):
        p.error(f"--executor-out: build only, inside {ASSETS}")
    if a.kernel_cache is not None and not a.kernel_cache.resolve().is_relative_to(EAGLE_N_ROOT.resolve()):
        raise ValueError(f"kernel-cache must stay inside {EAGLE_N_ROOT}")
    return a


def mounts(sdk: Path) -> list[str]:
    real = EAGLE_N_ROOT.resolve()
    out = [
        "--mount", f"type=bind,source={real},target={real}",
        "--mount", f"type=bind,source={real},target={EAGLE_N_ROOT}",
    ]  # fmt: skip
    if not ROOT.is_relative_to(real):
        out += ["--mount", f"type=bind,source={ROOT},target={ROOT}"]
    out += [
        "--mount", f"type=bind,source={sdk},target={sdk},readonly",
        "--mount", f"type=bind,source={sdk},target=/tt-metal,readonly",
    ]  # fmt: skip
    return out


def container_env(a, out: Path) -> dict[str, str]:
    view = out / "runtime/sdk_view"
    cache = a.kernel_cache.resolve() if a.kernel_cache is not None else out / "metal_cache"
    env = {
        "HOME": str(out),
        "TMPDIR": str(out / "tmp"),
        "XDG_CACHE_HOME": str(out / "cache"),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TT_METAL_HOME": str(view),
        "TT_METAL_CACHE": str(cache),
        "ARCH_NAME": "blackhole",
        "TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS": "BR,NC,TR,ER",
        "LD_LIBRARY_PATH": ":".join(
            str(a.sdk / "build_Release_distributed" / n) for n in ("ttnn", "tt_metal", "lib", "tt_stl")
        ),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "EAGLE_N_SDK_ROOT": str(a.sdk),  # build_native.sh; the SDK is mounted at this path
    }
    if a.executor_out is not None:
        env["EAGLE_N_EXECUTOR_OUT"] = str(a.executor_out.resolve())
    # l1 (bring-up of new graphs): a device wait longer than 60 s raises 'TIMEOUT: device timeout, potential hang
    # detected' inside the SDK instead of spinning forever. 'run' (SDV-BMT measurement) keeps its environment unchanged.
    if a.mode == "l1":
        env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "60"
        if a.sync_each_op:
            env["EAGLE_N_SYNC_EACH_OP"] = "1"
    return env


def entry(a, out: Path) -> list[str]:
    python = str(ROOT / ".venv/bin/python")
    if a.mode == "build":
        return ["--entrypoint", "/bin/bash", IMAGE, str(ROOT / "tools/eagle_n/build_native.sh")]
    if a.mode == "build-runner":
        return ["--entrypoint", "/bin/bash", IMAGE, str(ROOT / "tools/eagle_n/build_runner.sh")]
    if a.mode == "l1":
        args = [python, IMAGE, "-B", str(ROOT / "tools/eagle_n/l1_native.py"), "--range", a.target]
        args += ["--out", str(out / "l1"), "--rss-probe-slots", str(a.rss_probe_slots)]
        if a.dut:
            args += ["--dut", str(a.dut.resolve())]
        if a.workload:
            args += ["--workload", str(a.workload.resolve())]
        if a.preflight:
            args += ["--preflight"]
        if a.plan_override:
            args += ["--plan-override", str(a.plan_override.resolve())]
        if a.golden_extra:
            args += ["--golden-extra", str(a.golden_extra)]
        return ["--entrypoint", *args]
    if a.mode == "describe":
        import yaml

        dut = Path(a.target) if a.target else ROOT / "configs/duts/eagle-n-r18.yaml"
        opts = yaml.safe_load(dut.read_text())["runtime_options"]
        args = json.dumps({"executor": opts["executor"], "executor_sha256": opts["executor_sha256"]})
        return ["--entrypoint", str(ROOT / "native/build/sdv-bmt-native-eagle-n"), IMAGE, "describe", args]
    args = [a.mode, str(Path(a.target).resolve())]
    if a.mode == "run":
        args += ["--out", str(out / "bundles")]
    return ["--entrypoint", python, IMAGE, "-B", "-m", "sdv_bmt.cli.main", *args]


def main():
    a = parse_args()
    if a.mode == "run":
        precheck(a.target, a.evidence)
    if a.mode == "l1":
        precheck_device()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = a.evidence / (a.mode + "_" + stamp)
    out.mkdir(parents=True, mode=0o700)
    view = out / "runtime/sdk_view"
    (view / "generated").mkdir(parents=True)
    for name in ("tt_metal", "ttnn", "runtime", "tt_stl", "tools"):
        (view / name).symlink_to(a.sdk / name, target_is_directory=True)
    (view / "build").symlink_to(a.sdk / "build_Release_distributed", target_is_directory=True)
    env = container_env(a, out)
    for name in ("tmp", "cache", "metal_cache"):
        (out / name).mkdir()
    name = "sdv-eagle-n-" + stamp.lower().replace("_", "-")
    cmd = ["sudo", "-n", "docker", "run", "--rm", "--init", "--name", name, "--read-only", "--network", "none"]
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL", "--ulimit", "core=0:0"]
    cmd += mounts(a.sdk) + ["--workdir", str(ROOT)]
    huge = out / "hugepages"
    mounted = False
    code = 1
    try:
        if a.mode in ("run", "l1"):
            free = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages")
            if not free.exists() or int(free.read_text()) < 1:
                raise ValueError("existing 1GiB hugepage pool has no free page; no system reservation is changed")
            huge.mkdir(mode=0o700)
            options = f"pagesize=1G,uid={os.getuid()},gid={os.getgid()},mode=0700"
            subprocess.run(
                ["sudo", "-n", "mount", "-t", "hugetlbfs", "-o", options, "hugetlbfs", str(huge)], check=True
            )
            mounted = True
            cmd += ["--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1:-1", "--device", f"{DEVICE}:{DEVICE}:rwm"]
            cmd += ["--mount", f"type=bind,source={huge},target=/dev/hugepages-1G"]
            if a.mode == "run":
                # SDV-BMT writes the whole canonical-input staging file to /dev/shm when writable. The docker default
                # /dev/shm is 64 MB and a RAM tmpfs would need about 12 GiB; bind a private disk directory instead.
                shm = out / "shm"
                shm.mkdir(mode=0o700)
                cmd += ["--mount", f"type=bind,source={shm},target=/dev/shm"]
        for k, v in env.items():
            cmd += ["--env", k + "=" + v]
        cmd += entry(a, out)
        record = {"command": cmd, "device_exposed": a.mode in ("run", "l1"), "reset": False}
        (out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        with (a.evidence / "COMMANDS.log").open("a") as log:
            log.write(json.dumps({"utc": stamp, "command": cmd, "folder": str(out)}) + "\n")
        with (out / "run.log").open("w") as log:
            try:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.wall_seconds).returncode
            except subprocess.TimeoutExpired:
                code = 124
                # A container holding the NPU is NOT killed: killing it with kernels resident can leave the Tensix
                # cores wedged (the next open failed, a board reset was needed). Stop and report.
                if a.mode not in ("run", "l1"):
                    stop = ["sudo", "-n", "docker", "stop", "--time", "5", name]
                    subprocess.run(stop, capture_output=True, timeout=20)
        print((out / "run.log").read_text(errors="replace")[-6000:])
    finally:
        if mounted:
            inspect = ["sudo", "-n", "docker", "inspect", "--format", "{{.State.Running}}", name]
            probe = subprocess.run(inspect, capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip() == "true":
                status = {
                    "exit_code": code,
                    "device_exposed": True,
                    "npu_reset": False,
                    "container_left_running": name,
                    "private_mount_removed": False,
                    "action": "STOP: NPU process hung and was NOT killed; do not kill it; report it",
                }
                (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
                raise RuntimeError(f"container {name} still running (NOT killed); private hugepage mount preserved")
            for f in huge.iterdir():
                if (
                    f.is_symlink()
                    or not f.is_file()
                    or not (f.name == "tenstorrent" or f.name.startswith("device_0_channel_"))
                ):
                    raise RuntimeError("unexpected private hugepage file; preserve mount")
                f.unlink()
            subprocess.run(["sudo", "-n", "umount", str(huge)], check=True)
        shm = out / "shm"
        left = []
        if shm.is_dir():  # remove a leftover SDV-BMT staging file; keep small spec files as evidence
            for f in shm.glob("sdv-bmt-staging-*.bin"):
                left.append(f.name)
                f.unlink()
        status = {
            "exit_code": code,
            "removed_leftover_staging": left,
            "device_exposed": a.mode in ("run", "l1"),
            "npu_reset": False,
            "private_mount_removed": not mounted or not os.path.ismount(huge),
        }
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("Evidence:", out)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as e:
        print("REJECTED BEFORE DEVICE ACCESS:", str(e), file=sys.stderr)
        raise SystemExit(2)
