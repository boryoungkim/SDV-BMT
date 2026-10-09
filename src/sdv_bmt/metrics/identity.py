"""Output identity diagnostic (``bmt.output.identity_rate``, design v03 H.6).

The reference must be designated before the run (external golden output);
it is embedded in run_spec.json, so evaluation reads the bundle only.  The
first nominal run is never used as an implicit reference.  Identity is a
determinism/corruption diagnostic, not an accuracy metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..contracts.ids import digest_of, sha256_hex
from ..contracts.records import CompletionRecord, InputItem, RequestRecord
from ..contracts.specs import OutputReferenceEntry, SpecError, parse_strict
from ..contracts.status import MetricStatus, ReferenceKind, RequestState
from ..observation.reader import read_output_blob
from .portable import Computed, ratio


@dataclass(frozen=True)
class IdentityResult:
    computed: Computed
    notes: tuple[str, ...]


def _entries(reference: dict[str, Any]) -> list[OutputReferenceEntry]:
    entries = [parse_strict(OutputReferenceEntry, e, "output_reference.entries") for e in reference.get("entries", [])]
    if digest_of(entries) != reference.get("entries_digest"):
        raise SpecError("output_reference", "embedded entries do not match entries_digest")
    return entries


def output_identity(
    run_dir: Path,
    reference: dict[str, Any],
    folded: dict[int, tuple[RequestRecord, Optional[CompletionRecord]]],
    items: tuple[InputItem, ...],
) -> IdentityResult:
    if reference.get("kind") == ReferenceKind.NONE.value:
        return IdentityResult(
            Computed(MetricStatus.NOT_APPLICABLE, None, 0, "no_reference_designated"),
            ("no output reference was designated in the plan before the run",),
        )
    try:
        by_index = {e.item_index: e for e in _entries(reference)}
    except SpecError as exc:
        return IdentityResult(Computed(MetricStatus.INVALID, None, 0, "reference_integrity"), (str(exc),))

    compared = matches = without_reference = 0
    for request, completion in folded.values():
        if request.state is not RequestState.COMPLETED or completion is None or completion.output_ref is None:
            continue
        ref = completion.output_ref
        blob = read_output_blob(run_dir, ref)
        if len(blob) != ref.length or sha256_hex(blob) != ref.sha256:
            return IdentityResult(
                Computed(MetricStatus.INVALID, None, compared, "output_blob_digest_mismatch"),
                (f"request {request.request_id}: stored output does not match its output_ref",),
            )
        item = items[request.item_indices[0]]
        entry = by_index.get(item.item_index)
        if entry is None:
            without_reference += 1
            continue
        compared += 1
        if entry.source_item_id == item.source_item_id and entry.output_sha256 == ref.sha256:
            matches += 1
    notes = (
        f"reference kind={reference.get('kind')} sha256={reference.get('sha256')}",
        f"completed outputs without a reference entry: {without_reference}",
        "late-completion outputs are excluded (request not COMPLETED)",
    )
    if compared == 0:
        return IdentityResult(Computed(MetricStatus.EMPTY, None, 0, "no_comparable_outputs"), notes)
    return IdentityResult(Computed(MetricStatus.VALID, ratio(matches, compared), compared), notes)
