"""Factor identity and comparison keys (design v03 D.2 / J.3, M-3; design update
"Factor Isolation, Realized-Condition Validity").

Every identity is recorded as a named component so that a guard can name the
identity that changed.  ``COMMON_COMPONENTS`` are kept fixed by every contrast;
``FACTOR_COMPONENTS`` are the input and stress factors.  Small identities are
stored as readable values, large ones as SHA-256 digests; a difference in a
digest component (``workload_identity``, ``input_pipeline``,
``stress_axis_digest``, ...) is localized further by comparing the resolved
plans recorded in ``run_spec.json``.

Keys (summaries of the design identity; the guards live in ``validity.py``):

* ``key_stress_contrast``   — everything except ``stress_level``/``stress_level_value``
* ``key_scenario_contrast`` — everything except ``input_selection_digest``/``items_digest``
  (``null`` unless the input is a driving scenario)
* ``key_interaction``       — everything except selection, items and level
* ``cell_digest``           — everything

``stress_level_table`` (the level -> value mapping of the axis) is fixed, so a
level name always means the same intensity inside one contrast.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts.ids import digest_of, to_jsonable
from ..contracts.records import (
    ComparisonKeyRecord,
    DUTCapabilities,
    DUTDescriptor,
    InputDescriptor,
    InputPipelineRecord,
    input_pipeline_digest,
)
from ..contracts.specs import DUTConfig, ReplayPolicy, RunConfig, WorkloadSpec, workload_identity_digest
from ..contracts.status import BENCHMARK_LEVEL, ContrastType, ExecutionKind, InputKind, RunClockDomain

KEY_SCHEMA_VERSION = 2

COMMON_COMPONENTS = (
    "benchmark_level",
    "execution_kind",
    "dut_config",
    "dut_descriptor",
    "dut_capabilities",
    "workload_identity",
    "workload_task",
    "model_artifact",
    "precision_batch",
    "postprocess_codec",
    "preprocess_semantics_id",
    "input_pipeline",
    "measurement_boundary_id",
    "run_clock_domain",
    "replay",
    "admission_policy",
    "timeout_drain",
    "output_capture",
    "telemetry",
    "host_placement",
    "software_revision",
    "metric_definition",
    "output_reference",
    "realtime_harness",
)

FACTOR_COMPONENTS = (
    "input_kind",
    "input_family_digest",
    "input_selection_digest",
    "items_digest",
    "stress_axis_digest",
    "stress_level_table",
    "stress_level",
    "stress_level_value",
)

DESIGN_COMPONENTS = COMMON_COMPONENTS + FACTOR_COMPONENTS

# The only components a contrast may vary; everything else must be identical.
CONTRAST_VARIES: dict[ContrastType, tuple[str, ...]] = {
    ContrastType.STRESS: ("stress_level", "stress_level_value"),
    ContrastType.SCENARIO: ("input_selection_digest", "items_digest"),
    ContrastType.INTERACTION: ("input_selection_digest", "items_digest", "stress_level", "stress_level_value"),
}


def common_fixed_identity(
    *,
    dut_config: DUTConfig,
    dut_descriptor: DUTDescriptor,
    dut_capabilities: DUTCapabilities,
    workload: WorkloadSpec,
    input_pipeline: InputPipelineRecord,
    measurement_boundary_id: str,
    replay: ReplayPolicy,
    run: RunConfig,
    software_revision: str,
    execution_kind: ExecutionKind,
    metric_catalog_version: str,
    output_reference: dict[str, Any],
    run_clock_domain: RunClockDomain = RunClockDomain.VIRTUAL,
    telemetry: Any = "none",
    host_placement: Any = "virtual",
    realtime_harness: Any = None,
) -> dict[str, Any]:
    """Fixed identity of a run.  Stage 1A virtual runs keep the defaults; Stage 1R
    realtime runs pass the monotonic clock, the adapter's instrumentation
    profile (``telemetry``) and the realized host placement."""
    parts = {
        "benchmark_level": BENCHMARK_LEVEL,
        "execution_kind": execution_kind.value,
        # split so that a config, adapter (self-description) or capability change is told apart
        "dut_config": {k: v for k, v in to_jsonable(dut_config).items() if k != "dut_config_id"},
        "dut_descriptor": dut_descriptor,
        "dut_capabilities": dut_capabilities,
        "workload_identity": workload_identity_digest(workload),
        "workload_task": workload.task,
        "model_artifact": workload.artifact_sha256,
        "precision_batch": {"precision": workload.precision, "batch": workload.batch},
        "postprocess_codec": {"postprocess": workload.postprocess, "codec": workload.codec},
        "preprocess_semantics_id": input_pipeline.preprocess_semantics_id,
        "input_pipeline": input_pipeline_digest(input_pipeline),
        "measurement_boundary_id": measurement_boundary_id,
        "run_clock_domain": run_clock_domain.value,
        "replay": replay,
        "admission_policy": {
            "max_inflight": run.max_inflight,
            "queue_capacity": run.queue_capacity,
            "on_full": run.on_full,
            "on_request_error": run.on_request_error,
        },
        "timeout_drain": {"request_timeout_ns": run.request_timeout_ns, "drain_ns": run.drain_ns},
        "output_capture": run.output_capture.value,
        "telemetry": to_jsonable(telemetry),
        "host_placement": to_jsonable(host_placement),
        "software_revision": software_revision,
        "metric_definition": metric_catalog_version,
        # designated before the run; its location (path) is a label, not identity
        "output_reference": {k: v for k, v in output_reference.items() if k != "path"},
        # Stage 1R: issuer wait policy and pre-measure conditioning; null for virtual runs
        "realtime_harness": to_jsonable(realtime_harness),
    }
    assert tuple(parts) == COMMON_COMPONENTS
    return parts


def comparison_keys(
    *,
    common_parts: dict[str, Any],
    input_descriptor: InputDescriptor,
    stress_axis_digest: str,
    stress_level_table: dict[str, float],
    stress_level: str,
    stress_level_value: Optional[float],
) -> ComparisonKeyRecord:
    common_parts = to_jsonable(common_parts)
    components = {
        "common_fixed_identity": digest_of(common_parts),
        "common_parts": common_parts,
        "input_kind": input_descriptor.input_kind.value,
        "input_family_digest": input_descriptor.input_family_digest,
        "input_selection_digest": input_descriptor.input_selection_digest,
        "items_digest": input_descriptor.items_digest,
        "stress_axis_digest": stress_axis_digest,
        "stress_level_table": dict(stress_level_table),
        "stress_level": stress_level,
        "stress_level_value": stress_level_value,
    }
    identity = {**common_parts, **{name: components[name] for name in FACTOR_COMPONENTS}}

    def key(kind: str, excluded: tuple[str, ...]) -> str:
        return digest_of({"key": kind, **{k: v for k, v in identity.items() if k not in excluded}})

    scenario = input_descriptor.input_kind is InputKind.DRIVING_SCENARIO
    return ComparisonKeyRecord(
        schema_version=KEY_SCHEMA_VERSION,
        components=components,
        cell_digest=key("cell", ()),
        key_stress_contrast=key("stress_contrast", CONTRAST_VARIES[ContrastType.STRESS]),
        key_scenario_contrast=key("scenario_contrast", CONTRAST_VARIES[ContrastType.SCENARIO]) if scenario else None,
        key_interaction=key("interaction", CONTRAST_VARIES[ContrastType.INTERACTION]) if scenario else None,
    )


def identity_components(comparison_keys_record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a manifest ``comparison_keys`` record into {component: value}."""
    components = comparison_keys_record["components"]
    flat = dict(components["common_parts"])
    flat.update({name: components[name] for name in FACTOR_COMPONENTS if name in components})
    return flat
