"""Raw event schema (design v03 I.1), versions 1 and 2.

Version 1 is the Stage 1A virtual-clock schema.  Version 2 (Stage 1R realtime
bundles, ``run_clock_domain=monotonic``) adds the ``device_observation`` kind:
per-request device-domain timing and backend observations reported by a
polling adapter.  One bundle uses one version.

Every line of ``events.jsonl`` and ``stress.jsonl`` carries the common fields
``schema_version, kind, seq, t_run_ns, run_clock_domain``.  ``seq`` is one
counter shared by both files, so their union is a single total order.  Events
contain no run ID, so repeated runs of one plan produce identical bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.ids import to_jsonable

EVENT_SCHEMA_VERSION = 1
REALTIME_EVENT_SCHEMA_VERSION = 2
EVENT_SCHEMA_VERSIONS = (EVENT_SCHEMA_VERSION, REALTIME_EVENT_SCHEMA_VERSION)
COMMON_FIELDS = ("schema_version", "kind", "seq", "t_run_ns", "run_clock_domain")
STRESS_KIND = "stress"

# kind -> (required fields, optional fields)
EVENT_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "phase": (frozenset({"phase", "edge"}), frozenset({"outcome", "error", "details"})),
    "cleanup": (frozenset({"step", "outcome"}), frozenset({"error", "details"})),
    "issued": (frozenset({"request_id", "item_indices", "phase", "t_sched", "t_issue"}), frozenset()),
    "dropped": (frozenset({"request_id", "item_indices", "phase", "t_sched", "reason"}), frozenset()),
    "not_issued": (frozenset({"request_id", "item_indices", "phase", "t_sched", "reason"}), frozenset()),
    "submit_begin": (frozenset({"request_id"}), frozenset()),
    "submit_end": (frozenset({"request_id"}), frozenset()),
    "completed": (frozenset({"request_id", "t_complete", "t_output_ready", "output_ref"}), frozenset()),
    "error": (frozenset({"request_id", "error_phase", "error"}), frozenset({"t_complete"})),
    "timed_out": (frozenset({"request_id", "timeout_ns"}), frozenset()),
    "late_completion": (
        frozenset({"request_id", "result", "t_complete"}),
        frozenset({"t_output_ready", "output_ref", "error_phase", "error"}),
    ),
    "device_observation": (frozenset({"request_id"}), frozenset({"timing", "timing_method", "observations"})),
    "run_finished": (
        frozenset({
            "execution_status", "execution_reason", "measurement_status",
            "measurement_reason", "request_counts", "stress_status",
        }),
        frozenset(),
    ),
    STRESS_KIND: (
        frozenset({
            "step", "outcome", "provider", "stress_axis_digest", "stress_level", "requested",
            "achieved", "independent_evidence", "verification", "error", "details",
        }),
        frozenset(),
    ),
}

# kinds that exist only from the given schema version on
_KIND_MIN_VERSION = {"device_observation": REALTIME_EVENT_SCHEMA_VERSION}

REQUEST_KINDS = frozenset(
    {"issued", "dropped", "not_issued", "submit_begin", "submit_end", "completed",
     "error", "timed_out", "late_completion"}
)


class EventSchemaError(ValueError):
    pass


def make_event(kind: str, seq: int, t_run_ns: int, clock_domain: str, fields: Mapping[str, Any],
               schema_version: int = EVENT_SCHEMA_VERSION) -> dict[str, Any]:
    if kind not in EVENT_FIELDS:
        raise EventSchemaError(f"unknown event kind {kind!r}")
    if schema_version not in EVENT_SCHEMA_VERSIONS or schema_version < _KIND_MIN_VERSION.get(kind, 0):
        raise EventSchemaError(f"kind {kind!r} is not part of event schema version {schema_version}")
    required, optional = EVENT_FIELDS[kind]
    keys = set(fields)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise EventSchemaError(f"{kind}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    event: dict[str, Any] = {
        "schema_version": schema_version,
        "kind": kind,
        "seq": seq,
        "t_run_ns": t_run_ns,
        "run_clock_domain": clock_domain,
    }
    event.update(to_jsonable(dict(fields)))
    return event


def validate_event(event: object) -> list[str]:
    """Schema problems of one parsed event line (empty list = valid)."""
    if not isinstance(event, Mapping):
        return ["event is not an object"]
    problems = [f"missing common field {f}" for f in COMMON_FIELDS if f not in event]
    if problems:
        return problems
    if event["schema_version"] not in EVENT_SCHEMA_VERSIONS:
        problems.append(f"unsupported schema_version {event['schema_version']!r}")
    kind = event["kind"]
    if kind not in EVENT_FIELDS:
        return problems + [f"unknown kind {kind!r}"]
    if event["schema_version"] in EVENT_SCHEMA_VERSIONS and event["schema_version"] < _KIND_MIN_VERSION.get(kind, 0):
        problems.append(f"kind {kind!r} is not part of event schema version {event['schema_version']}")
    for name in ("seq", "t_run_ns"):
        if type(event[name]) is not int or event[name] < 0:
            problems.append(f"{name} must be a non-negative integer")
    required, optional = EVENT_FIELDS[kind]
    keys = set(event) - set(COMMON_FIELDS)
    if required - keys:
        problems.append(f"{kind}: missing {sorted(required - keys)}")
    if keys - required - optional:
        problems.append(f"{kind}: unknown {sorted(keys - required - optional)}")
    for name in sorted(_NON_NEGATIVE_INT_FIELDS & keys):
        if type(event[name]) is not int or event[name] < 0:
            problems.append(f"{kind}: {name} must be a non-negative integer")
    if "item_indices" in keys:
        indices = event["item_indices"]
        if not (isinstance(indices, list) and indices
                and all(type(i) is int and i >= 0 for i in indices)):
            problems.append(f"{kind}: item_indices must be a non-empty list of non-negative integers")
    return problems


# Request-event fields used in metric arithmetic; their types are checked on read.
_NON_NEGATIVE_INT_FIELDS = frozenset(
    {"request_id", "t_sched", "t_issue", "t_complete", "t_output_ready", "timeout_ns"}
)
