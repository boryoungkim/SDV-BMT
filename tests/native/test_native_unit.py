"""Native measurement plane: C++ unit tests (loop semantics on a simulated clock, event serialization,
canonical JSON, SHA-256) and byte equality of the native output pipeline with the Python reference."""

from __future__ import annotations

import json
import random
import struct
import subprocess

import pytest

from sdv_bmt.contracts.records import InputItem
from sdv_bmt.contracts.specs import WorkloadSpec, parse_strict
from sdv_bmt.runner.plan import load_yaml_strict
from sdv_bmt.workload.reference import ReferenceConvNetWorkload
from tests._helpers import CONFIGS, ZERO_SHA


def test_cpp_unit_suite_passes(native_build):
    proc = subprocess.run([str(native_build / "sdv-bmt-native-tests")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    last = proc.stdout.strip().splitlines()[-1]
    assert last.startswith("checks ") and last.endswith(" failures 0"), proc.stdout
    names = {line.split()[1] for line in proc.stdout.splitlines() if line.startswith("ok ")}
    assert {"drop_new", "timeout_late_completion_holds_slot", "inclusive_drain_cap", "inclusive_cap_after_preemption",
            "not_issued", "arrival_before_a_late_observed_completion",
            "late_wake_times_out_before_the_observed_completion", "completion_and_timeout_at_the_microsecond_boundary",
            "recorded_outcomes_follow_due_time_order_under_stalls", "asap", "errors",
            "output_bytes_match_python_encoding", "canonical_json", "sha256"} <= names


def _logit_vectors() -> list[list[int]]:
    rng = random.Random(20261008)
    vectors = [[0] * 10, [1] * 10, [-1] * 10, [2**63 - 1] * 10, [-(2**63)] * 10,
               [5, 5, 1, 0, 0, 0, 0, 0, 0, 5],  # tie: first maximum wins
               [-(2**63), 2**63 - 1, 0, 0, 0, 0, 0, 0, 0, 2**63 - 1]]
    vectors += [[rng.randint(-(2**40), 2**40) for _ in range(10)] for _ in range(300)]
    vectors += [[rng.randint(-3, 3) for _ in range(10)] for _ in range(100)]
    return vectors


@pytest.mark.parametrize("task", ["reference-conformance", "unicode tâche \"quoted\"\\path"])
def test_native_output_pipeline_bytes_equal_the_python_reference(native_build, task):
    spec = parse_strict(WorkloadSpec, load_yaml_strict(CONFIGS / "workloads" / "reference-convnet-v1.yaml"))
    import dataclasses
    workload = ReferenceConvNetWorkload(dataclasses.replace(spec, task=task))
    vectors = _logit_vectors()
    lines = "".join(json.dumps({"item_index": i, "task": task, "logits": v}) + "\n" for i, v in enumerate(vectors))
    proc = subprocess.run([str(native_build / "sdv-bmt-native-fake"), "encode-reference"], input=lines,
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    native = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(native) == len(vectors)
    for i, (v, row) in enumerate(zip(vectors, native)):
        item = InputItem(i, f"x-{i}", "p", ZERO_SHA, None, None, None, None)
        expected = workload.encode(workload.postprocess(struct.pack("<10q", *v), item))
        assert row["blob"].encode("utf-8") == expected, (i, v)
