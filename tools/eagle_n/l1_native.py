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
Correctness anchors (per plan variant, see GOLDENS):
- r18: the standalone golden image (r18_trial input) must give the validated label hash before and after the
  items (same trace => also a determinism check).
- r50: no validated NPU hash exists for the BF16 / bfloat8_b plans yet. The standalone golden input
  (frankfurt_000000_000294, canonical FP32 NCHW) must reach >= MIN_AGREEMENT pixel agreement with the ORT FP32
  labels, and every golden run must give the same label hash (determinism).
With --range START:0 only the golden checks run (L0: one image, prepare/run time and RSS).
Partial results can be merged exactly (counts are summed) with tools/eagle_n/l1_merge.py.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, sys, time
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
R50_GOLDEN = Path('/home/bmt/caontd/eagle_n_porting/deeplab_r50/golden')
R101_GOLDEN = Path('/home/bmt/caontd/eagle_n_porting/deeplab_r101/golden')
GOLDENS = {
    'r18': {'input': GOLDEN_PPM, 'expect_sha256': VALIDATED},
    # INTERNAL L0 anchor for R50 (2026-11-16): ORT FP32 labels of the standalone golden image; 0.99 is a sanity
    # floor (R18 BF16 vs ORT on its golden image: 0.9987), not G15. Hashes from golden/SHA256SUMS.json.
    'r50': {'input': R50_GOLDEN / 'input_fp32_nchw.npy',
            'input_sha256': 'adcef1a0c12d8ab5f0be7d2d2a56cff7b80fde719a97d6e734dd95d3044507a6',
            'reference': R50_GOLDEN / 'ort_labels_uint8.npy',
            'reference_sha256': '45b7b7b158a524445f1f7f8f57fe18a713ed8894d8b1b3b96e92ab1209216c51',
            'min_agreement': 0.99},
    # R101 (2026-11-17): same standalone golden image (input hash identical to R50), ORT FP32 labels of the R101 H04 ONNX
    # (eagle_n_porting/deeplab_r101/golden/SHA256SUMS.json). Same INTERNAL 0.99 sanity floor.
    'r101': {'input': R101_GOLDEN / 'input_fp32_nchw.npy',
             'input_sha256': 'adcef1a0c12d8ab5f0be7d2d2a56cff7b80fde719a97d6e734dd95d3044507a6',
             'reference': R101_GOLDEN / 'ort_labels_uint8.npy',
             'reference_sha256': '7ac360539cce8c652da26ec535030f00fb4a3b2a78b52cc7df895990b3be6a80',
             'min_agreement': 0.99},
}


def npy_payload(path: Path, descr: str, shape: tuple, want_sha: str) -> bytes:
    """Raw C-order payload of a version-1 .npy file, after checking its SHA-256, dtype and shape (no numpy)."""
    data = path.read_bytes()
    if sha(data) != want_sha:
        raise SystemExit(f'golden digest mismatch: {path}')
    if data[:6] != b'\x93NUMPY' or data[6] != 1:
        raise SystemExit(f'not a v1 .npy file: {path}')
    n = int.from_bytes(data[8:10], 'little')
    header = data[10:10 + n].decode('latin1')
    if f"'descr': '{descr}'" not in header or "'fortran_order': False" not in header or f"'shape': {shape}" not in header:
        raise SystemExit(f'unexpected .npy header {header.strip()} in {path}')
    return data[10 + n:]


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
    ap.add_argument('--preflight', action='store_true',
                    help='L1 preflight only: the executor runs the graph under ttnn NO_DISPATCH (no program sent to the cores) '
                         'and reports the peak L1 per core of every plan op; no staging, no run, no warm-up, no trace')
    ap.add_argument('--golden-extra', type=int, default=0,
                    help='non-R18 only: K extra identical golden replays after before/repeat (same-image latency); default 0')
    ap.add_argument('--plan-override', type=Path,
                    help='preflight only: use this plan file instead of the workload plan (e.g. the known-OOM validation plan)')
    a = ap.parse_args()
    if a.plan_override and not a.preflight:
        raise SystemExit('--plan-override is allowed only with --preflight')
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
    if not (0 <= start and count >= 0 and start + count <= len(lines)):
        raise SystemExit('range outside index')
    variant = prog['variant']
    if variant not in GOLDENS:
        raise SystemExit(f'no golden anchor for variant {variant}')
    gcfg = GOLDENS[variant]
    if variant == 'r18':
        w, h, rgb = parse_ppm(GOLDEN_PPM.read_bytes())
        golden_chw = normalize_rgb(rgb, w, h)
        reference = None
    else:
        golden_chw = npy_payload(gcfg['input'], '<f4', (1, 3, 1024, 2048), gcfg['input_sha256'])
        reference = npy_payload(gcfg['reference'], '|u1', (1, 1, 1024, 2048), gcfg['reference_sha256'])

    lib = ctypes.CDLL(opts['executor'])
    get = lib.sdv_eagle_executor_v1
    get.restype = ctypes.POINTER(API)
    api = get().contents
    if api.abi != 1 or not api.real_device:
        raise SystemExit('unexpected executor ABI / mock executor')
    err = ctypes.create_string_buffer(4096)
    plan_path = str(a.plan_override) if a.plan_override else prog['plan']
    spec = {'program': {'plan_path': plan_path, 'weights_path': prog['weights']},
            'backend': {'options': {'staging_budget_bytes': opts['staging_budget_bytes'],
                                    'prepare_host_budget_bytes': opts['prepare_host_budget_bytes'],
                                    'trace_region_bytes': opts['trace_region_bytes'], 'trial_unqualified': False}}}
    report = {'status': 'FAILED', 'range': [start, count], 'index_sha256': sha(index_bytes),
              'executor_sha256': opts['executor_sha256'], 'plan_sha256': prog['plan_sha256'],
              'weights_sha256': prog['weights_sha256'], 'variant': variant,
              'plan_precision': json.loads(Path(prog['plan']).read_text()).get('precision'),
              'plan_graph': json.loads(Path(prog['plan']).read_text()).get('graph'),
              'describe': json.loads(api.describe().decode()),
              'mode': 'non-trial: prepare = warm graph + trace capture; run = H2D + trace replay + blocking D2H',
              'scope': 'L1 offline numeric check (labels + exact confusion counts); not an SDV-BMT run'}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    if a.preflight:
        pf = a.out / 'preflight.json'
        os.environ['EAGLE_N_PREFLIGHT_OUT'] = str(pf)   # read by the executor at prepare
        report.update(mode='L1 preflight: graph under ttnn NO_DISPATCH; no program enqueued, no allocation, no device read/write',
                      scope='host-side L1 check before any device execution; not an inference',
                      plan_path=plan_path, plan_override_sha256=sha(Path(plan_path).read_bytes()) if a.plan_override else None)
        t0 = time.perf_counter()
        handle = api.prepare(json.dumps(spec).encode(), err, len(err))
        report['preflight_s'] = time.perf_counter() - t0
        if not handle:
            report['error'] = err.value.decode(errors='replace')
        else:
            if not api.destroy(handle, err, len(err)):
                report['destroy_error'] = err.value.decode(errors='replace')
        if pf.exists():
            r = json.loads(pf.read_text())
            report['preflight'] = {k: r[k] for k in ('l1_bank_size', 'l1_num_banks', 'limit_fraction', 'ops_checked',
                                                     'max_single_l1_buffer_per_core', 'max_single_op_index', 'ops_hard_over_limit',
                                                     'max_peak_l1_per_core', 'max_peak_op_index', 'ops_cumulative_over_limit',
                                                     'error', 'passed')}
            report['preflight_sha256'] = sha(pf.read_bytes())
            top = sorted((o for o in r['ops'] if 'peak_l1_per_core' in o), key=lambda o: -o['peak_l1_per_core'])[:8]
            for o in top:
                print('PREFLIGHT_TOP cumulative %8d B %5.1f%% | single buffer %8d B | CB set %7d B | %s %s %s' % (
                      o['peak_l1_per_core'], 100 * o['fraction_of_bank'], o['max_single_l1_buffer_per_core'],
                      o['max_cb_set_per_core'], o['kind'], o.get('launch', {}).get('config', '-'), o['output']), flush=True)
        report['status'] = 'PASSED' if report.get('preflight', {}).get('passed') and 'error' not in report else 'FAILED'
        (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
        print('PREFLIGHT ' + json.dumps({'status': report['status'], **report.get('preflight', {})}), flush=True)
        return 0 if report['status'] == 'PASSED' else 1
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
        labels, ms = infer(golden_chw)
        row = {'tag': tag, 'labels_sha256': sha(labels), 'run_ms': ms}
        prev = report.get('golden_checks', [])
        same = all(r['labels_sha256'] == row['labels_sha256'] for r in prev)
        if reference is None:
            row['equals_validated'] = sha(labels) == gcfg['expect_sha256']
            ok_row = row['equals_validated']
        else:
            agree = sum(x == y for x, y in zip(labels, reference)) / PIX
            row.update(agreement_vs_ort_fp32=agree, min_agreement=gcfg['min_agreement'], same_as_previous_golden=same)
            ok_row = agree >= gcfg['min_agreement'] and same
            if tag == 'before':
                (a.out / 'golden_labels.u8').write_bytes(labels)
        report.setdefault('golden_checks', []).append(row)
        print('GOLDEN ' + json.dumps(row), flush=True)
        return ok_row

    ok = golden('before')
    if reference is not None:   # r18 report layout unchanged (qualify_native / l1_merge read it)
        ok = golden('repeat') and ok    # second replay of the captured trace: determinism + warm timing
        for k in range(a.golden_extra):  # optional extra identical replays (latency of the same image), default 0
            ok = golden(f'extra{k + 1}') and ok
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
        chw = golden_chw
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
    miou, ious = total.result() if count else (None, None)
    s = sorted(times or [r['run_ms'] for r in report['golden_checks']])
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
