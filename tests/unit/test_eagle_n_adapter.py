"""Eagle-N interface tests; no NPU or vendor SDK required.

Tests that need the private Eagle-N assets (under /opt/sdv_assets/eagle_n, tools/eagle_n/README.md) skip as NOT RUN
when they are absent. Everything else uses synthetic files in a temporary directory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sdv_bmt.contracts.specs import DUTConfig, WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import frame_period_ns
from sdv_bmt.contracts.status import CompletionModel
from sdv_bmt.dut.eagle_n import adapter as eagle
from sdv_bmt.dut.eagle_n.adapter import EagleNDUT, EagleNOptions, EagleNProgram
from sdv_bmt.dut.eagle_n.executor_source import BUILD_RECORD_SCHEMA, build_record_path, executor_source_sha256
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.deeplab import DeepLabCityscapesWorkload
from tests._helpers import REPO_ROOT

ROOT = REPO_ROOT
EAGLE_N_ROOT = Path("/opt/sdv_assets/eagle_n")
ASSETS = EAGLE_N_ROOT / "eagle_n_integration_assets"
PORTING = EAGLE_N_ROOT / "eagle_n_porting"
# Eagle-N configurations: name -> (variant, config tag, assets dir). R-18 only, BF16 plan.
CONFIGS = {"r18": ("r18", "", "r18")}
ONNX = {
    "r18": PORTING / "deeplab_r18/models/deeplabv3plus_r18.onnx",
    "r50": PORTING / "deeplab_r50/models/deeplabv3plus_r50.onnx",
}
R18_PLAN = ASSETS / "r18/r18.plan.json"


def _paths(name):
    v, t, d = CONFIGS[name]
    return dict(
        capture=ROOT / f"configs/experiments/deeplabv3plus-{v}-eagle-n{t}-capture.yaml",
        baseline=ROOT / f"configs/experiments/deeplabv3plus-{v}-eagle-n{t}-baseline.yaml",
        workload=ROOT / f"configs/workloads/deeplabv3plus-{v}-cityscapes-v1-eagle-n{t}.yaml",
        dut=ROOT / f"configs/duts/eagle-n-{v}{t}.yaml",
        receipt=ASSETS / d / "native_qualification.json",
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def config():
    return load_yaml_strict(ROOT / "configs/duts/eagle-n-r18.yaml")


# ----------------------------------------------------------------------------------------------- options and configs
@pytest.mark.parametrize(
    "key,value",
    [
        ("device_id", 1),
        ("device_id", True),
        ("trace_region_bytes", 0),
        ("executor", "relative/executor.so"),
        ("executor_sha256", "f" * 63),
        ("executor_source_sha256", "F" * 64),
        ("staging_budget_bytes", 1),
    ],
)
def test_options_reject_invalid(config, key, value):
    data = copy.deepcopy(config["runtime_options"])
    data[key] = value
    with pytest.raises(ValueError):
        parse_strict(EagleNOptions, data)


def test_options_require_the_executor_source_pin(config):
    data = copy.deepcopy(config["runtime_options"])
    data.pop("executor_source_sha256")
    with pytest.raises(ValueError):
        parse_strict(EagleNOptions, data)


def test_unknown_option_is_error(config):
    config["runtime_options"]["invented_override"] = True
    with pytest.raises(ValueError):
        EagleNDUT(parse_strict(DUTConfig, config))


def test_configs_hold_no_host_paths_and_use_the_fixed_root():
    for f in (ROOT / "configs").rglob("*eagle-n*.yaml"):
        text = f.read_text()
        assert "/home/" not in text, f
        for line in text.splitlines():
            if line.strip().startswith(
                ("executor:", "lock_path:", "plan:", "weights:", "compile_record:", "qualification:")
            ):
                assert line.split(":", 1)[1].strip().startswith(str(ASSETS)), (f, line)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_plugin_preserves_canonical_workload(name):
    model = CONFIGS[name][0]
    base = load_yaml_strict(ROOT / f"configs/workloads/deeplabv3plus-{model}-cityscapes-v1.yaml")
    new = load_yaml_strict(_paths(name)["workload"])
    a, b = copy.deepcopy(base), copy.deepcopy(new)
    for d in (a, b):
        for k in ("workload_id", "precision"):
            d.pop(k)
        d["options"].pop("device_program")
    assert a == b
    w = DeepLabCityscapesWorkload(parse_strict(WorkloadSpec, new))
    assert w.input_contract().contract_id == "cityscapes-rgb-norm-chw2048x1024-f32"
    p = parse_strict(EagleNProgram, new["options"]["device_program"])
    assert p.task_output_bytes == 2097152 and new["precision"] == "bf16-conv-fp32dst"


def test_one_executor_and_one_source_pin_for_every_model():
    opts = [load_yaml_strict(_paths(n)["dut"])["runtime_options"] for n in CONFIGS]
    assert len({o["executor"] for o in opts}) == 1
    assert len({o["executor_sha256"] for o in opts}) == 1
    assert {o["executor_source_sha256"] for o in opts} == {executor_source_sha256()}


def test_no_unqualified_or_legacy_eagle_n_configs():
    names = sorted(f.name for f in (ROOT / "configs").rglob("*eagle-n*"))
    assert not [n for n in names if any(k in n for k in ("bfp8w", "provisional", "asap-t60", ".yaml.in"))], names
    assert not [n for n in names if "r50" in n or "r101" in n], names  # Eagle-N is configured for R-18 only
    for f in (ROOT / "configs").rglob("*eagle-n*"):
        assert "fp32" not in f.read_text().lower().replace("fp32dst", ""), f


@pytest.mark.parametrize("name", sorted(CONFIGS))
@pytest.mark.parametrize("stage", ["capture", "baseline"])
def test_plans_are_the_gpu_plans_with_one_eagle_n_timeout(name, stage):
    model = CONFIGS[name][0]
    eagle_plan = load_yaml_strict(_paths(name)[stage])
    gpu = load_yaml_strict(ROOT / f"configs/experiments/deeplabv3plus-{model}-{stage}.yaml")
    assert eagle_plan["run"]["request_timeout_ns"] == eagle_plan["run"]["drain_ns"] == 60_000_000_000
    assert eagle_plan["dut"] == f"../duts/{_paths(name)['dut'].name}"
    assert eagle_plan["workload"] == f"../workloads/{_paths(name)['workload'].name}"
    if stage == "capture":
        assert eagle_plan["replay"]["mode"] == "asap"
        assert eagle_plan.get("evaluation", {}).get("output_reference") is None
    else:
        ref = eagle_plan["evaluation"]["output_reference"]
        assert "eagle-n" in ref["path"] and "gpu-capture" not in ref["path"]
        assert _sha(_paths(name)["baseline"].parent / ref["path"]) == ref["sha256"]
        gpu["evaluation"].pop("output_reference")
        eagle_plan["evaluation"].pop("output_reference")
        # the Eagle-N baseline rate of the model, deadline = its frame period (tools/eagle_n/configure.py)
        m = _module("eagle_configure_rate", "tools/eagle_n/configure.py")
        assert eagle_plan["replay"]["rate_hz"] == m.BASELINE_RATE_HZ[model]
        assert eagle_plan["evaluation"]["deadline"] == {
            "policy": "frame_period", "period_ns": frame_period_ns(m.BASELINE_RATE_HZ[model])
        }
        for d in (eagle_plan, gpu):
            d["replay"].pop("rate_hz")
            d["evaluation"]["deadline"].pop("period_ns")
    for d in (eagle_plan, gpu):
        for k in ("experiment_id", "workload", "dut"):
            d.pop(k)
        d["run"].pop("request_timeout_ns")
        d["run"].pop("drain_ns")
        d["realtime"].pop("cpu_affinity")
    assert eagle_plan == gpu


def test_configure_renders_the_committed_configs():
    """configure.py is the source of the configs: rendering with the committed pins gives the committed files."""
    m = _module("eagle_configure", "tools/eagle_n/configure.py")
    for name, (model, _, _) in CONFIGS.items():
        dut = load_yaml_strict(_paths(name)["dut"])
        work = load_yaml_strict(_paths(name)["workload"])["options"]["device_program"]
        base = load_yaml_strict(_paths(name)["baseline"])
        pins = {
            "executor_sha256": dut["runtime_options"]["executor_sha256"],
            "executor_source_sha256": dut["runtime_options"]["executor_source_sha256"],
            "plan_sha256": work["plan_sha256"],
            "weights_sha256": work["weights_sha256"],
            "compile_record_sha256": work["compile_record_sha256"],
            "qualification_sha256": work["qualification_sha256"],
            "golden_sha256": base["evaluation"]["output_reference"]["sha256"],
        }
        for path, (header, data) in m.render(model, pins, base["realtime"]["cpu_affinity"]).items():
            assert path.read_text() == header + yaml.safe_dump(data, sort_keys=False), path


def test_run_script_uses_the_committed_plans():
    s = (ROOT / "tools/eagle_n/run_eagle_n.sh").read_text()
    assert "  r18)" in s and "r50" not in s and "r101" not in s
    assert "CAPTURE=$E/$ID-capture.yaml" in s and "BASELINE=$E/$ID-baseline.yaml" in s
    assert "provisional" not in s and "asap-t60" not in s and "/home/" not in s


# ----------------------------------------------------- executor, build record and receipt (synthetic)
def _synthetic(tmp_path, model="r18", source=None, receipt_overrides=None, qualified=True):
    """DUT and workload documents whose files are small synthetic stand-ins in tmp_path."""
    executor = tmp_path / "libeagle_n_ttnn.so"
    executor.write_bytes(b"synthetic executor")
    src = source or executor_source_sha256()
    build_record_path(executor).write_text(
        json.dumps({"schema": BUILD_RECORD_SCHEMA, "executor_sha256": _sha(executor), "executor_source_sha256": src})
    )
    work = load_yaml_strict(_paths("r18")["workload"])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "sdv-bmt-eagle-n-ttnn-plan/1",
                "onnx_sha256": work["artifact_sha256"],
                "variant": model,
                "precision": "bf16",
                "ops": [],
            }
        )
    )
    weights = tmp_path / "weights.f32"
    weights.write_bytes(bytes(64))
    record = tmp_path / "compile_record.json"
    record.write_text(json.dumps({"plan_sha256": _sha(plan), "weights_sha256": _sha(weights)}))
    receipt = tmp_path / "native_qualification.json"
    body = {
        "passed": True,
        "variant": model,
        "executor_sha256": _sha(executor),
        "executor_source_sha256": src,
        "plan_sha256": _sha(plan),
        "weights_sha256": _sha(weights),
        **(receipt_overrides or {}),
    }
    receipt.write_text(json.dumps(body))
    work["options"]["device_program"].update(
        plan=str(plan), plan_sha256=_sha(plan), weights=str(weights), weights_sha256=_sha(weights),
        compile_record=str(record), compile_record_sha256=_sha(record), qualification=str(receipt),
        qualification_sha256=_sha(receipt) if qualified else None,
    )  # fmt: skip
    dut = load_yaml_strict(ROOT / "configs/duts/eagle-n-r18.yaml")
    dut["runtime_options"].update(
        executor=str(executor),
        executor_sha256=_sha(executor),
        executor_source_sha256=src,
        lock_path=str(tmp_path / "lock"),
    )
    return dut, work


def _backend_spec(dut, work):
    return EagleNDUT(parse_strict(DUTConfig, dut)).backend_spec(parse_strict(WorkloadSpec, work))


def test_consistent_executor_build_record_and_receipt_are_accepted(tmp_path, monkeypatch):
    for name in eagle.DIAGNOSTIC_ENV:
        monkeypatch.delenv(name, raising=False)
    dut, work = _synthetic(tmp_path)
    spec = _backend_spec(dut, work)
    assert spec["program"]["qualification_path"] == work["options"]["device_program"]["qualification"]


def test_executor_built_from_another_source_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path, source="0" * 64)
    with pytest.raises(ValueError, match="executor source of this checkout differs"):
        _backend_spec(dut, work)


def test_build_record_of_another_executor_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path)
    executor = Path(dut["runtime_options"]["executor"])
    executor.write_bytes(b"another executor")
    dut["runtime_options"]["executor_sha256"] = _sha(executor)
    with pytest.raises(ValueError, match="executor_sha256 is not the SHA-256"):
        _backend_spec(dut, work)


def test_missing_build_record_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path)
    build_record_path(dut["runtime_options"]["executor"]).unlink()
    with pytest.raises(ValueError, match="build record .* missing"):
        _backend_spec(dut, work)


def test_unpinned_receipt_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path, qualified=False)
    with pytest.raises(ValueError, match="qualification NOT RUN"):
        _backend_spec(dut, work)


def test_receipt_that_differs_from_the_workload_pin_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path)
    Path(work["options"]["device_program"]["qualification"]).write_text("{}")
    with pytest.raises(ValueError, match="differs from the workload pin"):
        _backend_spec(dut, work)


@pytest.mark.parametrize(
    "overrides",
    [{"executor_source_sha256": "0" * 64}, {"executor_source_sha256": None}, {"passed": False}, {"variant": "r50"}],
)
def test_receipt_for_another_source_or_failed_is_refused(tmp_path, overrides):
    dut, work = _synthetic(tmp_path, receipt_overrides=overrides)
    with pytest.raises(ValueError, match="qualification receipt stale"):
        _backend_spec(dut, work)


@pytest.mark.parametrize("name", ["EAGLE_N_PREFLIGHT_OUT", "EAGLE_N_SYNC_EACH_OP"])
def test_diagnostic_environment_is_refused_in_benchmark_runs(tmp_path, monkeypatch, name):
    dut, work = _synthetic(tmp_path)
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="diagnostic environment"):
        _backend_spec(dut, work)


def test_workload_precision_must_match_plan_precision(tmp_path):
    dut, work = _synthetic(tmp_path)
    work["precision"] = "fp32"
    with pytest.raises(ValueError, match="precision declaration mismatch"):
        _backend_spec(dut, work)


def test_non_bf16_plan_is_refused(tmp_path):
    dut, work = _synthetic(tmp_path)
    plan = Path(work["options"]["device_program"]["plan"])
    plan.write_text(plan.read_text().replace('"bf16"', '"bfp8w"'))
    dp = work["options"]["device_program"]
    dp["plan_sha256"] = _sha(plan)
    record = Path(dp["compile_record"])
    record.write_text(json.dumps({"plan_sha256": dp["plan_sha256"], "weights_sha256": dp["weights_sha256"]}))
    dp["compile_record_sha256"] = _sha(record)
    with pytest.raises(ValueError, match="Unsupported plan precision"):
        _backend_spec(dut, work)


# ------------------------------------------------------------------------------------------------ declarations
def test_measurement_declares_sync_cast_and_no_fabricated_device_time(config):
    d = EagleNDUT(parse_strict(DUTConfig, config))
    m = d.measurement_declaration()
    assert m["device_timing"] is None and m["prepare_input_h2d"] == "included"
    assert "synchronous" in m["instrumentation"]["submit_semantics"]
    assert "collect" in m["instrumentation"]["output_conversion"]
    assert d.capabilities().completion_model == CompletionModel.POLLING and not d.capabilities().device_timing
    assert len(d.realization_evidence(None)) == 8


def test_unqualified_trial_is_explicit_and_declares_jit(config):
    config["runtime_options"]["trial_unqualified"] = True
    dut = EagleNDUT(parse_strict(DUTConfig, config))
    text = dut.measurement_declaration()["instrumentation"]["submit_semantics"]
    assert "DIAGNOSTIC UNQUALIFIED" in text and "JIT/allocation" in text
    for variant in ("r50", "r101"):
        work = load_yaml_strict(_paths("r18")["workload"])
        work["artifact_sha256"] = eagle.H04[variant]
        work["options"]["device_program"]["variant"] = variant
        with pytest.raises(ValueError, match="restricted to R18"):
            dut.backend_spec(parse_strict(WorkloadSpec, work))


@pytest.mark.parametrize("value", [1, "yes", None])
def test_unqualified_trial_requires_boolean(config, value):
    config["runtime_options"]["trial_unqualified"] = value
    with pytest.raises(ValueError):
        EagleNDUT(parse_strict(DUTConfig, config))


def test_sdk_dependency_drift_blocks_before_native_describe(tmp_path, monkeypatch):
    dut, _ = _synthetic(tmp_path)
    executor = dut["runtime_options"]["executor"]
    library = tmp_path / "libtt_metal.so"
    library.write_bytes(b"sdk library")
    Path(executor + ".sdk.json").write_text(
        json.dumps({"sdk_commit": "0" * 40, "libraries_sha256": {str(library): "0" * 64}})
    )
    monkeypatch.setattr(
        eagle, "describe_native", lambda *a, **k: pytest.fail("must reject SDK drift before the native process")
    )
    with pytest.raises(ValueError, match="SDK dependency drift"):
        EagleNDUT(parse_strict(DUTConfig, dut)).describe()


def test_exporter_and_adapter_agree_on_plan_precisions():
    m = _module("eagle_export2", "tools/eagle_n/export_plan.py")
    assert m.PRECISIONS == eagle.PLAN_PRECISIONS
    assert set(eagle.PRECISION_DECLARATION) == {p for v in eagle.PLAN_PRECISIONS.values() for p in v} == {"bf16"}


# ----------------------------------------------------------------------------------------------------- launcher
def _launcher(monkeypatch):
    m = _module("eagle_container", "tools/eagle_n/in_container.py")

    def forbidden(*a, **kw):
        raise AssertionError("No device/subprocess before qualification")

    monkeypatch.setattr(m.subprocess, "run", forbidden)
    return m


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_actual_launcher_refuses_unqualified_graph_before_any_subprocess(monkeypatch, name):
    m = _launcher(monkeypatch)
    with pytest.raises(ValueError, match="qualification NOT RUN"):
        m.precheck(_paths(name)["capture"])


def _synthetic_plan(tmp_path, dut, work):
    (tmp_path / "dut.yaml").write_text(yaml.safe_dump(dut))
    (tmp_path / "workload.yaml").write_text(yaml.safe_dump(work))
    plan = load_yaml_strict(_paths("r18")["capture"])
    plan.update(
        input=os.path.relpath(ROOT / "configs/inputs/cityscapes-val500.yaml", tmp_path),
        dut="dut.yaml",
        workload="workload.yaml",
    )
    plan["stress"]["spec"] = os.path.relpath(ROOT / "configs/stress/fake-virtual-axis.yaml", tmp_path)
    path = tmp_path / "capture.yaml"
    path.write_text(yaml.safe_dump(plan))
    return path


def test_actual_launcher_refuses_stale_qualification_before_any_subprocess(tmp_path, monkeypatch):
    dut, work = _synthetic(tmp_path, receipt_overrides={"executor_source_sha256": "0" * 64})
    m = _launcher(monkeypatch)
    with pytest.raises(ValueError, match="qualification stale or failed"):
        m.precheck(_synthetic_plan(tmp_path, dut, work), tmp_path)


def test_actual_launcher_refuses_a_receipt_that_differs_from_the_pin(tmp_path, monkeypatch):
    dut, work = _synthetic(tmp_path)
    work["options"]["device_program"]["qualification_sha256"] = "0" * 64
    m = _launcher(monkeypatch)
    with pytest.raises(ValueError, match="differs from the workload pin"):
        m.precheck(_synthetic_plan(tmp_path, dut, work), tmp_path)


# ---------------------------------------------------------------------------------- exporter (needs the private assets)
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_native_plan_identity_and_bindings(name):
    model = CONFIGS[name][0]
    work = load_yaml_strict(_paths(name)["workload"])
    p = work["options"]["device_program"]
    if not Path(p["plan"]).exists():
        pytest.skip("NOT RUN: Eagle-N assets not present on this host")
    plan = json.loads(Path(p["plan"]).read_text())
    blob = Path(p["weights"]).read_bytes()
    assert plan["onnx_sha256"] == work["artifact_sha256"]
    assert plan["logical_convs"] == {"r18": 37, "r50": 70, "r101": 121}[model]
    for op in plan["ops"]:
        if op["kind"] == "Conv":
            for key in ("weight", "bias"):
                x = op[key]
                assert hashlib.sha256(blob[x["offset"] : x["offset"] + x["bytes"]]).hexdigest() == x["sha256"]
    assert plan["ops"][-1]["kind"] == "Cast" and plan["ops"][-2]["kind"] == "ArgMax"


def test_h04_exporter_fails_before_parsing_unknown_artifact(tmp_path):
    m = _module("eagle_export", "tools/eagle_n/export_plan.py")
    p = tmp_path / "wrong.onnx"
    p.write_bytes(b"not-the-assigned-model")
    with pytest.raises(ValueError, match="H04"):
        m.convert(p, "r18", "bf16")


def test_tuned_plans_need_the_r18_envelope():
    if not all(p.exists() for p in (*ONNX.values(), R18_PLAN)):
        pytest.skip("NOT RUN: Eagle-N assets not present on this host")
    m = _module("eagle_export3", "tools/eagle_n/export_plan.py")
    r18, _ = m.convert(ONNX["r18"], "r18", "bf16")
    assert all("launch" not in o for o in r18["ops"]) and "launch_envelope" not in r18
    with pytest.raises(ValueError, match="--envelope"):
        m.convert(ONNX["r50"], "r50", "bf16", "tuned")
    with pytest.raises(ValueError, match="--envelope"):
        m.convert(ONNX["r18"], "r18", "bf16", None, R18_PLAN)


def test_proven_survey_dominance_rules(tmp_path):
    m = _module("eagle_survey", "tools/eagle_n/proven_survey.py")
    assert m.shapes(["Tensor(shape=Shape([1, 1, 32768, 1024]))", "Shape([2048, 1024, 1, 1])"]) == [
        (1, 1, 32768, 1024),
        (2048, 1024, 1, 1),
    ]
    assert m.dominated(((1, 1, 512, 1024),), ((1, 1, 32768, 1024),))
    assert not m.dominated(((1, 1, 32768, 2048),), ((1, 1, 32768, 1024),))
    assert m.dominated(((512, 1024),), ((1, 1, 512, 1024),))  # right-aligned rank padding

    def pf(path, ops):
        path.write_text(json.dumps({"ops": ops}))
        return str(path)

    def call(n, *s):
        return {"n": n, "a": ["Shape([" + ", ".join(map(str, x)) + "])" for x in s]}

    proven = pf(
        tmp_path / "p.json",
        [
            {
                "index": 0,
                "kind": "Conv",
                "output": "a",
                "calls": [call("Matmul", (1, 1, 32768, 1024), (1, 1, 1024, 1024))],
            },
            {
                "index": 1,
                "kind": "Conv",
                "output": "b",
                "calls": [call("Matmul", (1, 1, 32768, 1024), (1, 1, 1024, 2048))],
            },
        ],
    )
    cand = pf(
        tmp_path / "c.json",
        [
            {
                "index": 0,
                "kind": "Conv",
                "output": "x",
                "calls": [call("Matmul", (1, 1, 32768, 1024), (1, 1, 1024, 2048))],
            }
        ],
    )
    import sys as _s

    for last, expect in ((None, 0), (0, 1)):  # op #1 is proven only if the proven run got past it
        suffix = f":{last}" if last is not None else ""
        _s.argv = ["s", "--candidate", cand, "--proven", proven + suffix, "--out", str(tmp_path / "o.json")]
        assert m.main() == expect


def test_r50_bf16_decomposition_is_proven_and_equivalent():
    # layer4 + head are rewritten into blocks of proven shapes; verify() is the symbolic equivalence proof
    if not all(p.exists() for p in (ONNX["r50"], R18_PLAN)):
        pytest.skip("NOT RUN: Eagle-N assets not present on this host")
    m = _module("eagle_export4", "tools/eagle_n/export_plan.py")
    plan, _ = m.convert(ONNX["r50"], "r50", "bf16", "tuned", R18_PLAN)
    convs = [o for o in plan["ops"] if o["kind"] == "Conv"]
    assert plan["logical_convs"] == 70 and plan["decomposition"]["executed_conv_ops"] == len(convs)
    assert all(o["launch"]["config"] in ("r18_tuned", "native_default") for o in convs)  # no unproven standalone path
    assert not any(o["kind"] == "Conv" and o["shape"][1] > 1024 for o in plan["ops"])  # no 2048-ch conv output remains
    recon, _ = m.convert(ONNX["r50"], "r50", "bf16", "tuned", R18_PLAN, True)

    def strip(o):
        return {**o, "launch": {k: v for k, v in o["launch"].items() if k != "reason"}} if "launch" in o else o

    assert [strip(o) for o in recon["ops"][:101]] == [strip(o) for o in plan["ops"][:101]]  # ops #0-#100: device-proven


def test_npu_user_check_needs_sudo(monkeypatch, tmp_path):
    """fuser exits 1 for 'no user' and a refused sudo exits 1 too: without sudo -n the check must stop."""
    m = _module("eagle_container_sudo", "tools/eagle_n/in_container.py")
    device = tmp_path / "bos0"
    device.write_text("")
    monkeypatch.setattr(m, "DEVICE", str(device))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "sudo: a password is required")

    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(ValueError, match="sudo -n unavailable"):
        m.npu_idle()
    assert calls == [["sudo", "-n", "true"]]  # fuser is not asked
