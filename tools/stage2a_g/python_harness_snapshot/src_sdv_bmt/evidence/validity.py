"""Contrast validity: Design, Realization and Measurement guards.

    Valid attributed contrast = Design valid AND Realization valid AND Measurement valid

Control what is planned, verify what is realized, localize what changed, and
do not attribute beyond the evidence.  Guard first, arithmetic second: these
functions compare identities and evidence recorded in run manifests (and in
evaluation manifests, when given) and never look at metric values.

* Design Guard — did the specification vary only the intended factor?  Every
  design component is compared, every difference is listed, and only the
  contrast's allowed components may differ.  The factor counts as varied only
  when its substance changed (``stress_level_value``, ``items_digest``), not
  its label.  An identity that is unknown (null) on both sides leaves the
  guard ``unresolved`` unless both runs are mock.
* Realization Guard — were the intended-fixed conditions realized equivalently,
  as far as the recorded evidence shows?  A recorded difference fails.  ``pass``
  needs every condition *observed*, with a value, on both sides and compared
  under a supported equivalence rule (Stage 1A: ``exact``, canonical JSON
  equality); ``not_applicable`` is never evidence of equivalence.  A condition
  that carries the intended factor (the stress itself, the scenario payload)
  is exempt from the cross-run equality check only when that factor varied,
  and each side must still show the factor was realized as planned.  Mock (or
  any non-actual) execution never passes: a configured condition and a
  realized condition are not assumed to be identical.
* Measurement Guard — same measurement boundary, timestamp semantics, output
  capture, telemetry, metric catalog and output reference; both runs have a
  valid measurement; and both evaluations used the same protocol and
  evaluator.  Without the evaluation manifests the guard is ``unresolved``.

A pairwise interaction result only checks that two cells belong to one
interaction family: an interaction needs all four cells of the 2x2 (design v03
J.2), so it is never established or attributed from a pair.

Failure to establish factor isolation is not a benchmark failure; it is
evidence that the requested attribution is not justified.
"""

from __future__ import annotations

from typing import Any, Optional

from ..contracts.ids import canonical_json_bytes, to_jsonable
from ..contracts.records import (
    CONTRAST_VALIDITY_SCHEMA_VERSION,
    ContrastValidityResult,
    DesignGuardResult,
    MeasurementGuardResult,
    RealizationEvidence,
    RealizationGuardResult,
)
from ..contracts.specs import parse_strict
from ..contracts.status import (
    NOMINAL_LEVEL,
    ContrastType,
    ContrastValidity,
    Coverage,
    EvidenceValidity,
    ExecutionKind,
    GuardStatus,
    InputKind,
    NominalCheck,
    RealizedCondition,
    VerificationStatus,
)
from .keys import CONTRAST_VARIES, DESIGN_COMPONENTS, identity_components

MEASUREMENT_COMPONENTS = (
    "measurement_boundary_id",  # timestamp semantics, preprocess/H2D inclusion, output-ready, input pipeline
    "run_clock_domain",
    "output_capture",
    "telemetry",
    "metric_definition",
    "output_reference",
)
EVIDENCE_METADATA = ("source", "semantic", "unit", "scope", "clock", "equivalence")
SUPPORTED_EQUIVALENCE = ("exact",)
_COMPARABLE_VALIDITY = {EvidenceValidity.OBSERVED.value, EvidenceValidity.SELF_REPORTED.value,
                        EvidenceValidity.VIRTUAL.value}

# The component whose change *is* the factor; a label-only change is not a variation.
FACTOR_SUBSTANCE = {
    ContrastType.STRESS: ("stress_level_value",),
    ContrastType.SCENARIO: ("items_digest",),
    ContrastType.INTERACTION: ("items_digest", "stress_level_value"),
}
# Realized conditions that carry a design factor, with the component they realize.
FACTOR_CONDITIONS = {
    RealizedCondition.STRESS_REALIZATION.value: "stress_level_value",
    RealizedCondition.INPUT_PAYLOAD.value: "items_digest",
}
# Components whose null value is defined (the nominal level has no value; a
# virtual-clock run has no realtime harness).
NULLABLE_COMPONENTS = ("stress_level_value", "realtime_harness")
INTERACTION_NOTE = ("pairwise interaction: checks only that both cells belong to one interaction family; "
                    "an interaction needs all four cells of the 2x2 (design v03 J.2)")


def _same(x: Any, y: Any) -> bool:
    """Exact equivalence: canonical JSON equality (``True``, ``1`` and ``1.0`` differ)."""
    return canonical_json_bytes(x) == canonical_json_bytes(y)


def _execution_kinds(manifest: dict[str, Any], ids: dict[str, Any]) -> set[Any]:
    return {manifest.get("execution_kind"), manifest.get("status", {}).get("execution_kind"),
            manifest.get("realization", {}).get("execution_kind"), ids.get("execution_kind")}


def design_guard(a: dict[str, Any], b: dict[str, Any], contrast: ContrastType) -> DesignGuardResult:
    allowed = CONTRAST_VARIES[contrast]
    present = [c for c in DESIGN_COMPONENTS if c in a and c in b]
    observed = tuple(c for c in present if not _same(a[c], b[c]))
    missing = tuple(f"{c}:missing" for c in DESIGN_COMPONENTS if c not in a or c not in b)
    violations = tuple(c for c in observed if c not in allowed) + missing
    varied = any(c in a and c in b and not _same(a[c], b[c]) for c in FACTOR_SUBSTANCE[contrast])
    unknown = tuple(c for c in present if c not in NULLABLE_COMPONENTS and a[c] is None and b[c] is None)
    both_mock = a.get("execution_kind") == b.get("execution_kind") == ExecutionKind.MOCK.value
    notes: list[str] = []
    if violations:
        status = GuardStatus.FAIL
    elif contrast is not ContrastType.STRESS and not (
        a.get("input_kind") == b.get("input_kind") == InputKind.DRIVING_SCENARIO.value
    ):
        status = GuardStatus.NOT_APPLICABLE
        notes.append(f"{contrast.value} contrasts require driving_scenario input on both sides")
    elif unknown and not both_mock:
        status = GuardStatus.UNRESOLVED  # an unknown identity cannot be shown to be the same
    else:
        status = GuardStatus.PASS
    if contrast is ContrastType.INTERACTION:
        notes.append(INTERACTION_NOTE)
    if not varied:
        notes.append("intended factor did not vary (replicate pair)")
    return DesignGuardResult(status, allowed, observed, violations, varied, tuple(notes), unknown)


def _protocol(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in evaluation.get("protocol", {}).items() if k != "deadline_source"}


def _evaluator(evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluator = evaluation.get("evaluator", {})
    return {
        "module": evaluator.get("module"),
        "catalog_version": evaluator.get("catalog_version"),
        "definition_versions": evaluator.get("definition_versions"),
        "source_tree_sha256": evaluator.get("code_identity", {}).get("source_tree_sha256"),
    }


def _differing(prefix: str, x: dict[str, Any], y: dict[str, Any]) -> list[str]:
    return [f"{prefix}.{k}" for k in sorted(set(x) | set(y))
            if k not in x or k not in y or not _same(x[k], y[k])]


def measurement_guard(manifest_a: dict[str, Any], manifest_b: dict[str, Any], a: dict[str, Any],
                      b: dict[str, Any], evaluations: Optional[tuple[dict[str, Any], dict[str, Any]]] = None,
                      ) -> MeasurementGuardResult:
    violations = [c if c in a and c in b else f"{c}:missing" for c in MEASUREMENT_COMPONENTS
                  if not (c in a and c in b and _same(a[c], b[c]))]
    for side, manifest, ids in (("run_a", manifest_a, a), ("run_b", manifest_b, b)):
        recorded = manifest["measurement_boundary"]["measurement_boundary_id"]
        if recorded != ids.get("measurement_boundary_id"):
            violations.append(f"{side}:boundary_record_inconsistent")
        status = manifest["status"]
        if status["execution_status"] != "completed" or status["measurement_status"] != "valid":
            violations.append(f"{side}:measurement_not_valid({status['execution_status']},"
                              f"{status['measurement_status']},{status['measurement_reason']})")
    notes: list[str] = []
    if evaluations is None:
        notes.append("evaluation manifests not given: evaluation protocol and evaluator identity not compared")
    else:
        eval_a, eval_b = evaluations
        for side, manifest, evaluation in (("run_a", manifest_a, eval_a), ("run_b", manifest_b, eval_b)):
            if evaluation.get("run_id") != manifest["run_id"]:
                violations.append(f"{side}:evaluation_of_another_run")
        violations += _differing("evaluation_protocol", _protocol(eval_a), _protocol(eval_b))
        violations += _differing("evaluator", _evaluator(eval_a), _evaluator(eval_b))
    identity = {
        "run_a": a.get("measurement_boundary_id"),
        "run_b": b.get("measurement_boundary_id"),
        "equal": a.get("measurement_boundary_id") == b.get("measurement_boundary_id"),
    }
    if violations:
        status = GuardStatus.FAIL
    elif evaluations is None:
        status = GuardStatus.UNRESOLVED
    else:
        status = GuardStatus.PASS
    return MeasurementGuardResult(status, identity, tuple(violations), tuple(notes))


def _evidence(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Strictly parsed evidence by condition; a malformed or duplicate record is rejected."""
    records: dict[str, dict[str, Any]] = {}
    for raw in manifest["realization"]["evidence"]:
        record = to_jsonable(parse_strict(RealizationEvidence, raw, "realization.evidence"))
        if record["condition"] in records:
            raise ValueError(f"run {manifest.get('run_id')}: duplicate realization evidence "
                             f"for {record['condition']!r}")
        records[record["condition"]] = record
    return records


def _factor_realized(condition: str, record: dict[str, Any], ids: dict[str, Any]) -> Optional[tuple[str, str]]:
    """None when the evidence shows the factor realized as planned on this side,
    else ("contradicted" | "not_established", reason)."""
    if record["validity"] not in _COMPARABLE_VALIDITY:
        return "not_established", f"factor_evidence_{record['validity']}"
    value = record["value"]
    if not isinstance(value, dict):
        return "not_established", "factor_evidence_unreadable"
    if condition == RealizedCondition.INPUT_PAYLOAD.value:
        if not _same(value.get("items_digest"), ids.get("items_digest")):
            return "contradicted", "realized_items_differ_from_plan"
    elif ids.get("stress_level") == NOMINAL_LEVEL:
        if value.get("nominal_check") == NominalCheck.FAILED.value:
            return "contradicted", "nominal_check_failed"
        if value.get("nominal_check") != NominalCheck.PASSED.value:
            return "not_established", "nominal_check_not_passed"
    else:
        if (value.get("verification") == VerificationStatus.FAILED.value
                or value.get("coverage") == Coverage.NONE.value):
            return "contradicted", "stress_not_realized"
        if not (value.get("verification") == VerificationStatus.VERIFIED.value
                and value.get("coverage") == Coverage.FULL.value):
            return "not_established", "stress_not_verified_over_the_full_window"
    if record["validity"] != EvidenceValidity.OBSERVED.value:
        return "not_established", "factor_evidence_not_independently_observed"
    return None


def realization_guard(manifest_a: dict[str, Any], manifest_b: dict[str, Any], ids_a: dict[str, Any],
                      ids_b: dict[str, Any], contrast: ContrastType) -> RealizationGuardResult:
    ev_a, ev_b = _evidence(manifest_a), _evidence(manifest_b)
    allowed = CONTRAST_VARIES[contrast]
    intended = tuple(
        condition for condition, component in FACTOR_CONDITIONS.items()
        if component in allowed and not _same(ids_a.get(component), ids_b.get(component))
    )
    available: list[str] = []
    mismatches: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for condition in (c.value for c in RealizedCondition):
        a, b = ev_a.get(condition), ev_b.get(condition)
        if a is None or b is None:
            unresolved.append({"condition": condition, "reason": "evidence_missing"})
            continue
        if condition in FACTOR_CONDITIONS:
            for side, record, ids in (("run_a", a, ids_a), ("run_b", b, ids_b)):
                problem = _factor_realized(condition, record, ids)
                if problem is not None:
                    kind, reason = problem
                    entry = {"condition": condition, "side": side, "reason": reason,
                             "validity": record["validity"], "value": record["value"]}
                    (mismatches if kind == "contradicted" else unresolved).append(entry)
            if condition in intended:
                continue  # carries the intended, varied factor: exempt from cross-run equality
        va, vb = a["validity"], b["validity"]
        if va != vb or va not in _COMPARABLE_VALIDITY:
            reason = "validity_differs" if va != vb else va  # not_observable / not_applicable
            unresolved.append({"condition": condition, "validity_a": va, "validity_b": vb, "reason": reason})
        elif any(not _same(a[k], b[k]) for k in EVIDENCE_METADATA):
            unresolved.append({"condition": condition, "reason": "evidence_semantics_differ"})
        elif a["equivalence"] not in SUPPORTED_EQUIVALENCE:
            unresolved.append({"condition": condition, "reason": "unsupported_equivalence_rule",
                               "equivalence": a["equivalence"]})
        elif not _same(a["value"], b["value"]):
            mismatches.append({"condition": condition, "reason": "value_differs", "validity": va,
                               "run_a": a["value"], "run_b": b["value"]})
        elif va == EvidenceValidity.OBSERVED.value:
            available.append(condition)
        else:
            unresolved.append({"condition": condition, "validity": va,
                               "reason": "equal_but_not_independently_observed"})
    kinds = _execution_kinds(manifest_a, ids_a) | _execution_kinds(manifest_b, ids_b)
    if mismatches:
        status, basis = GuardStatus.FAIL, "evidence_mismatch"
    elif kinds != {ExecutionKind.ACTUAL.value}:
        status = GuardStatus.NOT_OBSERVABLE
        basis = "mock_execution" if ExecutionKind.MOCK.value in kinds else "execution_kind_not_actual"
    elif unresolved:
        status, basis = GuardStatus.UNRESOLVED, "partial_evidence"
    else:
        status, basis = GuardStatus.PASS, "observed_evidence"
    return RealizationGuardResult(status, basis, tuple(available), tuple(mismatches), tuple(unresolved),
                                  intended)


def evaluate_contrast_validity(manifest_a: dict[str, Any], manifest_b: dict[str, Any],
                               contrast: ContrastType | str, *,
                               evaluations: Optional[tuple[dict[str, Any], dict[str, Any]]] = None,
                               ) -> ContrastValidityResult:
    """``evaluations`` are the two runs' ``eval_manifest.json`` contents, in run order."""
    contrast = ContrastType(contrast)
    a = identity_components(manifest_a["comparison_keys"])
    b = identity_components(manifest_b["comparison_keys"])
    design = design_guard(a, b, contrast)
    realization = realization_guard(manifest_a, manifest_b, a, b, contrast)
    measurement = measurement_guard(manifest_a, manifest_b, a, b, evaluations)
    statuses = (design.status, realization.status, measurement.status)
    if GuardStatus.FAIL in statuses:
        overall = ContrastValidity.INVALID
    elif design.status is GuardStatus.NOT_APPLICABLE:
        overall = ContrastValidity.NOT_APPLICABLE
    elif contrast is ContrastType.INTERACTION:
        overall = ContrastValidity.NOT_ESTABLISHED  # a pair cannot establish an interaction
    elif all(s is GuardStatus.PASS for s in statuses):
        overall = ContrastValidity.VALID
    else:
        overall = ContrastValidity.NOT_ESTABLISHED
    return ContrastValidityResult(
        schema_version=CONTRAST_VALIDITY_SCHEMA_VERSION,
        contrast_type=contrast,
        run_a=manifest_a["run_id"],
        run_b=manifest_b["run_id"],
        design_guard=design,
        realization_guard=realization,
        measurement_guard=measurement,
        overall_status=overall,
        # a valid contrast in which the intended factor did not vary is a replicate pair
        attribution_allowed=overall is ContrastValidity.VALID and design.intended_factor_varied,
    )
