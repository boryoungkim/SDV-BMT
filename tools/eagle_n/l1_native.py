#!/usr/bin/env python3
"""L1 offline numeric check, Eagle-N part: the native C++ executor on canonical SDV-BMT inputs.

Runs INSIDE the pinned BOS container (launch: tools/eagle_n/in_container.py l1 START:COUNT).
Same path as the backend: libeagle_n_ttnn.so via its C ABI, non-trial mode (prepare = warm graph +
trace capture; run = H2D of the staged input + trace replay + blocking D2H). One staging slot is
reused, so host memory holds one 12 MiB input regardless of COUNT. This is an offline numeric
check (guide L1), not an SDV-BMT run; the SDV-BMT qualification receipt is not involved and none is
written here.

Per item: canonical input (sdv_bmt DeepLab preprocessing of the val500 payload), labels (uint32 from
the device, checked < 19, narrowed to uint8 as collect() does), SHA-256, row-RLE blob, exact per-class
TP/FP/FN against the gtFine annotation (sdv_bmt ConfusionCounts), host-observed run() time.
Correctness anchors: the standalone golden image (r18_trial input) must give the validated label
hash before and after the items (same trace => also a determinism check).
Partial results can be merged exactly (counts are summed) with tools/eagle_n/l1_merge.py.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import yaml  # noqa: E402
from sdv_bmt.contracts import labelmap  # noqa: E402
from sdv_bmt.workload.deeplab import normalize_rgb  # noqa: E402
from sdv_bmt.workload.yolox import parse_ppm  # noqa: E402
from sdv_bmt.metrics.quality_segmentation import ConfusionCounts, ground_truth  # noqa: E402

W, H, PIX, IN_BYTES = 2048, 1024, 2048 * 1024, 25165824
VALIDATED = '621089306df561378d3c9aa56901184eda76c79ac46a156e7deaf6ce07935168'
GOLDEN_PPM = Path('/home/bmt/caontd/eagle_n_integration_assets/r18_trial/input.ppm')


def mem() -> dict:
    """VmRSS (resident now) and VmHWM (peak resident) of this process, MiB, from /proc/self/status."""
    out = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            k, v = line.split(':')
            out[k + '_MiB'] = round(int(v.split()[0]) / 1024, 1)
    return out


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class API(ctypes.Structure):
    _fields_ = [('abi', ctypes.c_uint32), ('real_device', ctypes.c_bool),
                ('describe', ctypes.CFUNCTYPE(ctypes.c_char_p)),
                ('prepare', ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t)),
                ('stage', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t)),
                ('run', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
                                         ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t)),
                ('destroy', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--range', required=True, help='START:COUNT over datasets/cityscapes-val500/index.jsonl')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dut', type=Path, default=ROOT / 'configs/duts/eagle-n-r18.yaml')
    ap.add_argument('--workload', type=Path, default=ROOT / 'configs/workloads/deeplabv3plus-r18-cityscapes-v1-eagle-n.yaml')
    ap.add_argument('--index', type=Path, default=ROOT / 'datasets/cityscapes-val500/index.jsonl')
    ap.add_argument('--rss-probe-slots', type=int, default=0,
                    help='after the items, stage the golden input into K extra slots (1..K) and record the RSS '
                         'growth per staged item (sizes the staging/RSS budgets for a full sdv-bmt run)')
    a = ap.parse_args()
    start, count = (int(x) for x in a.range.split(':'))
    a.out.mkdir(parents=True, exist_ok=True)
    opts = yaml.safe_load(a.dut.read_text())['runtime_options']
    prog = yaml.safe_load(a.workload.read_text())['options']['device_program']
    checks = {'executor': (opts['executor'], opts['executor_sha256']), 'plan': (prog['plan'], prog['plan_sha256']),
              'weights': (prog['weights'], prog['weights_sha256'])}
    for k, (path, h) in checks.items():
        if sha(Path(path).read_bytes()) != h:
            raise SystemExit(f'{k} digest mismatch: {path}')
    index_bytes = a.index.read_bytes()
    lines = index_bytes.decode().splitlines()
    if not (0 <= start and count > 0 and start + count <= len(lines)):
        raise SystemExit('range outside index')

    lib = ctypes.CDLL(opts['executor'])
    get = lib.sdv_eagle_executor_v1
    get.restype = ctypes.POINTER(API)
    api = get().contents
    if api.abi != 1 or not api.real_device:
        raise SystemExit('unexpected executor ABI / mock executor')
    err = ctypes.create_string_buffer(4096)
    spec = {'program': {'plan_path': prog['plan'], 'weights_path': prog['weights']},
            'backend': {'options': {'staging_budget_bytes': opts['staging_budget_bytes'],
                                    'prepare_host_budget_bytes': opts['prepare_host_budget_bytes'],
                                    'trace_region_bytes': opts['trace_region_bytes'], 'trial_unqualified': False}}}
    report = {'status': 'FAILED', 'range': [start, count], 'index_sha256': sha(index_bytes),
              'executor_sha256': opts['executor_sha256'], 'plan_sha256': prog['plan_sha256'],
              'weights_sha256': prog['weights_sha256'], 'describe': json.loads(api.describe().decode()),
              'mode': 'non-trial: prepare = warm graph + trace capture; run = H2D + trace replay + blocking D2H',
              'scope': 'L1 offline numeric check (labels + exact confusion counts); not an SDV-BMT run'}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    report['memory'] = {'start': mem()}
    t0 = time.perf_counter()
    handle = api.prepare(json.dumps(spec).encode(), err, len(err))
    report['prepare_s'] = time.perf_counter() - t0
    if not handle:
        raise SystemExit('prepare failed: ' + err.value.decode(errors='replace'))
    report['memory']['after_prepare'] = mem()
    print('MEM after_prepare ' + json.dumps(report['memory']['after_prepare']), flush=True)
    out = (ctypes.c_uint32 * PIX)()
    raw = memoryview(out).cast('B')

    def infer(chw: bytes):
        if len(chw) != IN_BYTES:
            raise SystemExit('canonical input size')
        if not api.stage(handle, 0, chw, IN_BYTES, err, len(err)):
            raise SystemExit('stage failed: ' + err.value.decode(errors='replace'))
        t = time.perf_counter_ns()
        if not api.run(handle, 0, out, PIX, err, len(err)):
            raise SystemExit('run failed: ' + err.value.decode(errors='replace'))
        ms = (time.perf_counter_ns() - t) / 1e6
        b = bytes(raw)
        if b[1::4].strip(b'\0') or b[2::4].strip(b'\0') or b[3::4].strip(b'\0'):
            raise SystemExit('label above 255 from device')
        labels = b[0::4]
        if labels.translate(None, bytes(range(19))):
            raise SystemExit('label outside 0..18 from device')
        return labels, ms

    def golden(tag):
        w, h, rgb = parse_ppm(GOLDEN_PPM.read_bytes())
        labels, ms = infer(normalize_rgb(rgb, w, h))
        row = {'tag': tag, 'labels_sha256': sha(labels), 'equals_validated': sha(labels) == VALIDATED, 'run_ms': ms}
        report.setdefault('golden_checks', []).append(row)
        print('GOLDEN ' + json.dumps(row), flush=True)
        return row['equals_validated']

    ok = golden('before')
    total = ConfusionCounts()
    times = []
    with (a.out / 'items.jsonl').open('w') as items, (a.out / 'labels.rle').open('wb') as blobs:
        for i in range(start, start + count):
            rec = json.loads(lines[i])
            payload = (a.index.parent / rec['payload_file']).read_bytes()
            if sha(payload) != rec['payload_sha256']:
                raise SystemExit(f'payload digest mismatch item {i}')
            w, h, rgb = parse_ppm(payload)
            labels, ms = infer(normalize_rgb(rgb, w, h))
            times.append(ms)
            gw, gh, gt = ground_truth(rec['annotation'])
            one = ConfusionCounts()
            one.add(labels, gt)
            total.add(labels, gt)
            blob = labelmap.encode(labels, W, H, i)
            off = blobs.tell()
            blobs.write(blob)
            items.write(json.dumps({'item_index': i, 'source_item_id': rec['source_item_id'],
                                    'labels_sha256': sha(labels), 'rle_offset': off, 'rle_length': len(blob),
                                    'run_ms': ms, 'tp': one.tp, 'fp': one.fp, 'fn': one.fn,
                                    'valid_pixels': one.pixels}) + '\n')
            items.flush()
            if (i - start) % 25 == 0:
                print(f'ITEM {i} run_ms={ms:.1f}', flush=True)
    ok = golden('after') and ok
    report['memory']['after_items'] = mem()
    if a.rss_probe_slots:
        w, h, rgb = parse_ppm(GOLDEN_PPM.read_bytes())
        chw = normalize_rgb(rgb, w, h)
        before = mem()
        for slot in range(1, a.rss_probe_slots + 1):
            if not api.stage(handle, slot, chw, IN_BYTES, err, len(err)):
                raise SystemExit('rss probe stage failed: ' + err.value.decode(errors='replace'))
        after = mem()
        grow = after['VmRSS_MiB'] - before['VmRSS_MiB']
        report['memory']['rss_probe'] = {'extra_slots': a.rss_probe_slots, 'before': before, 'after': after,
                                         'rss_growth_MiB': round(grow, 1),
                                         'per_staged_item_MiB': round(grow / a.rss_probe_slots, 2),
                                         'expected_per_item_MiB': 12.0}
        print('MEM rss_probe ' + json.dumps(report['memory']['rss_probe']), flush=True)
        ok = golden('after_rss_probe') and ok
    miou, ious = total.result()
    s = sorted(times)
    report.update(status='COMPLETED' if ok else 'GOLDEN_MISMATCH', items=count,
                  partial_miou_this_range_only=miou, counts={'tp': total.tp, 'fp': total.fp, 'fn': total.fn,
                                                             'valid_pixels': total.pixels},
                  run_ms_host_observed={'n': len(s), 'min': s[0], 'p50': s[len(s) // 2], 'max': s[-1],
                                        'note': 'host time around run(): H2D + trace replay + D2H; stage excluded; '
                                                'diagnostic, not an SDV-BMT DUT-service metric'})
    if not api.destroy(handle, err, len(err)):
        report['destroy_error'] = err.value.decode(errors='replace')
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('DONE ' + json.dumps({k: report[k] for k in ('status', 'items', 'partial_miou_this_range_only', 'run_ms_host_observed')}), flush=True)
    return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
