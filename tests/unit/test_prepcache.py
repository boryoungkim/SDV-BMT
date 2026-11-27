"""Prepared-input cache (runner/prepcache.py): verified reuse of Workload.preprocess results, opt-in by environment."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sdv_bmt.contracts.ids import file_sha256
from sdv_bmt.evidence.manifest import canonical_manifest_content, read_manifest
from sdv_bmt.runner.core import PreparedStore, prepare_inputs
from sdv_bmt.runner.prepcache import CACHE_ENV, KEY_METHOD, PreparedInputCache
from tests._helpers import load, run_plan

CONTEXT = {"source_tree_sha256": "0" * 64, "workload_implementation": "w", "preprocess": {"id": "p", "version": "1"},
           "input_contract_digest": "1" * 64}


def _item(index=0, payload="a" * 64):
    return {"item_index": index, "source_item_id": f"s{index}", "payload_sha256": payload}


def test_miss_put_hit_and_keys(tmp_path):
    cache = PreparedInputCache(tmp_path, CONTEXT)
    assert cache.get(_item()) is None
    cache.put(_item(), b"tensor-bytes")
    assert cache.get(_item()) == b"tensor-bytes"
    assert cache.stats == {"hits": 1, "misses": 1, "written": 1, "rejected": 0, "write_errors": 0,
                           "not_cacheable": 0}
    # another payload, another item index or another context (code, contract) is another key
    assert cache.get(_item(payload="b" * 64)) is None and cache.get(_item(index=1)) is None
    other = PreparedInputCache(tmp_path, {**CONTEXT, "source_tree_sha256": "2" * 64})
    assert other.get(_item()) is None
    rec = cache.record()
    assert rec["method"] == KEY_METHOD and len(rec["context_digest"]) == 64 and str(tmp_path) not in json.dumps(rec)


def test_damaged_entries_are_misses_and_get_rewritten(tmp_path):
    cache = PreparedInputCache(tmp_path, CONTEXT)
    cache.put(_item(), b"good")
    data_path, sum_path = cache._paths(cache.key(_item()))
    data_path.write_bytes(b"bad!")  # content changed: digest mismatch
    assert cache.get(_item()) is None and cache.stats["rejected"] == 1
    cache.put(_item(), b"good")
    assert cache.get(_item()) == b"good"
    sum_path.unlink()  # no sidecar: not trusted
    assert cache.get(_item()) is None and cache.stats["rejected"] == 2


def test_non_bytes_and_unwritable_never_fail(tmp_path):
    cache = PreparedInputCache(tmp_path, CONTEXT)
    cache.put(_item(), {"not": "bytes"})
    assert cache.stats["not_cacheable"] == 1 and cache.get(_item()) is None
    blocked = tmp_path / "file"
    blocked.write_text("x")
    bad = PreparedInputCache(blocked, CONTEXT)  # root is a file: every write fails
    bad.put(_item(), b"x")
    assert bad.stats["write_errors"] == 1 and bad.get(_item()) is None


@dataclass(frozen=True)
class _Item:
    item_index: int
    payload_sha256: str


class _Provider:
    def payload(self, index):
        return bytes([index]) * 4


class _Workload:
    def __init__(self):
        self.calls = 0

    def preprocess(self, payload, item):
        self.calls += 1
        return payload + b"!"


class _Adapter:
    def prepare_input(self, task_input, item):
        return ("prepared", task_input)


def test_prepare_inputs_reuses_cached_results(tmp_path):
    items = [_Item(i, f"{i:064x}") for i in range(3)]
    first, second = _Workload(), _Workload()
    a, b = PreparedStore(), PreparedStore()
    prepare_inputs(_Provider(), first, _Adapter(), items, a, PreparedInputCache(tmp_path, CONTEXT))
    cache = PreparedInputCache(tmp_path, CONTEXT)
    prepare_inputs(_Provider(), second, _Adapter(), items, b, cache)
    assert first.calls == 3 and second.calls == 0 and cache.stats["hits"] == 3
    assert [a.get(i) for i in range(3)] == [b.get(i) for i in range(3)]


def _prepare_event(run_dir):
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("phase") == "prepare_inputs" and e.get("edge") == "end":
            return e
    raise AssertionError("no prepare_inputs end event")


def test_runs_with_the_cache_record_it_and_produce_the_same_evidence(tmp_path, monkeypatch):
    rp = load("g1-dataset-fixed-rate-nominal")
    plain = run_plan(rp, tmp_path / "plain", run_id="run-plain")
    assert "prepared_input_cache" not in _prepare_event(plain.run_dir)["details"]  # unset: nothing changes

    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    cold = run_plan(rp, tmp_path / "cold", run_id="run-cold")
    warm = run_plan(rp, tmp_path / "warm", run_id="run-warm")
    c, w = (_prepare_event(r.run_dir)["details"]["prepared_input_cache"] for r in (cold, warm))
    n = _prepare_event(cold.run_dir)["details"]["prepared"]
    assert n > 0 and c["misses"] == n and c["written"] == n and c["hits"] == 0
    assert w["hits"] == n and w["misses"] == 0 and w["context_digest"] == c["context_digest"]
    # the same outputs and the same canonical manifest content as the run without the cache (raw event files differ
    # only by the cache record in the prepare_inputs event)
    for r in (cold, warm):
        assert file_sha256(r.run_dir / "outputs.bin") == file_sha256(plain.run_dir / "outputs.bin")
        assert file_sha256(r.run_dir / "input_items.jsonl") == file_sha256(plain.run_dir / "input_items.jsonl")
    base = canonical_manifest_content(read_manifest(plain.run_dir))
    for r in (cold, warm):
        other = canonical_manifest_content(read_manifest(r.run_dir))
        strip = lambda m: {k: v for k, v in m.items() if k != "files"}  # noqa: E731 - events.jsonl digest differs
        assert strip(other) == strip(base)
