#!/usr/bin/env python3
"""L1 offline numeric check, reference part: ONNX Runtime FP32 (CPU) on the same canonical inputs, compared
per image with the NPU labels from tools/eagle_n/l1_native.py. CPU only, no device, no sudo.

Three modes. All sdv_bmt metric code runs unchanged under the project Python (>= 3.10):
  --input-digests OUT.json  [project Python 3.11] SHA-256 of the canonical input tensor of each item
                            (sdv_bmt parse_ppm + normalize_rgb).
  --infer                   [Python with onnxruntime + numpy; may be older] ORT FP32 labels per item. Each
                            canonical input must hash equal to the --expect-input-digests value from the project
                            Python, so the ORT input is proven byte-identical. Writes ORT labels (labelmap-u8-rowrle,
                            sdv_bmt codec) + per-pixel agreement with the NPU labels. No metric is computed here.
  --score                   [project Python 3.11] decodes the ORT labels (SHA-256 checked), exact TP/FP/FN with
                            the unmodified sdv_bmt ConfusionCounts, ORT mIoU, and the summary report.
--infer is resumable: items already in OUT/items.jsonl are skipped.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sdv_bmt.contracts import labelmap  # noqa: E402
from sdv_bmt.workload.deeplab import normalize_rgb  # noqa: E402
from sdv_bmt.workload.yolox import parse_ppm  # noqa: E402

W, H = 2048, 1024
ONNX = Path('/home/bmt/caontd/eagle_n_porting/deeplab_r18/models/deeplabv3plus_r18.onnx')
ONNX_SHA = 'd7776aa683a18ca4e09305360f7d7f2857bac1a665dbbbc3aee2c60348450969'   # H04
CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
NPU_MIOU_L1 = 0.7675648837497185   # eagle_n_integration_evidence/l1_merged_20261114T183402Z
GPU_FP32_MIOU = 0.767566


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(index: Path, rec: dict, i: int) -> bytes:
    payload = (index.parent / rec['payload_file']).read_bytes()
    if sha(payload) != rec['payload_sha256']:
        raise SystemExit(f'payload digest mismatch item {i}')
    w, h, rgb = parse_ppm(payload)
    if (w, h) != (W, H):
        raise SystemExit(f'item {i}: size {w}x{h}')
    return normalize_rgb(rgb, w, h)


def read_blob(path: Path, off: int, length: int, i: int, want_sha: str) -> bytes:
    with path.open('rb') as f:
        f.seek(off)
        item, w, h, labels = labelmap.decode(f.read(length))
    if (item, w, h) != (i, W, H) or sha(labels) != want_sha:
        raise SystemExit(f'item {i}: label record inconsistent ({path})')
    return labels


def infer(a, lines, index_bytes, rng) -> int:
    import numpy as np
    import onnxruntime as ort
    if not (a.npu and a.expect_input_digests):
        raise SystemExit('--infer needs --npu and --expect-input-digests')
    d = json.loads(a.expect_input_digests.read_text())
    if d['index_sha256'] != sha(index_bytes):
        raise SystemExit('input digests: index digest differs')
    expected = {int(k): v for k, v in d['items'].items()}
    if any(i not in expected for i in rng):
        raise SystemExit('input digests do not cover the whole range')
    if file_sha(ONNX) != ONNX_SHA:
        raise SystemExit('ONNX H04 digest mismatch')
    npu = {}
    for p in a.npu:
        for line in (p / 'items.jsonl').read_text().splitlines():
            r = json.loads(line)
            npu[r['item_index']] = (p / 'labels.rle', r['rle_offset'], r['rle_length'], r['labels_sha256'])
    if any(i not in npu for i in rng):
        raise SystemExit('NPU labels missing for part of the range')
    items_path, rle_path = a.out / 'items.jsonl', a.out / 'ort_labels.rle'
    done = set()
    if items_path.exists():
        rows = [json.loads(x) for x in items_path.read_text().splitlines()]
        done = {r['item_index'] for r in rows}
        end = max((r['rle_offset'] + r['rle_length'] for r in rows), default=0)
        with rle_path.open('ab') as f:   # drop a blob written after the last completed item record
            f.truncate(end)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX), sess_options=opts, providers=['CPUExecutionProvider'])
    env = {'python': platform.python_version(), 'executable': sys.executable, 'onnxruntime': ort.__version__,
           'numpy': np.__version__, 'ort_file': ort.__file__, 'threads': a.threads, 'providers': ['CPUExecutionProvider'],
           'onnx': str(ONNX), 'onnx_sha256': ONNX_SHA, 'index_sha256': sha(index_bytes),
           'input_digests_python': d['python'], 'npu_parts': [str(p) for p in a.npu], 'range': [rng.start, len(rng)]}
    (a.out / 'infer_env.json').write_text(json.dumps(env, indent=1) + '\n')
    print('ENV ' + json.dumps(env), flush=True)
    with items_path.open('a') as items, rle_path.open('ab') as blobs:
        for i in rng:
            if i in done:
                continue
            rec = json.loads(lines[i])
            chw = canonical(a.index, rec, i)
            if sha(chw) != expected[i]:
                raise SystemExit(f'item {i}: canonical input differs from the project-Python digest')
            x = np.frombuffer(chw, dtype='<f4').reshape(1, 3, H, W)
            t = time.perf_counter()
            (lab,) = session.run(['labels'], {'input': x})
            ms = (time.perf_counter() - t) * 1e3
            ort_b = np.ascontiguousarray(lab.reshape(H, W), dtype=np.uint8).tobytes()
            if ort_b.translate(None, bytes(range(19))):
                raise SystemExit(f'item {i}: ORT label outside 0..18')
            path, off_n, len_n, npu_sha = npu[i]
            npu_b = read_blob(path, off_n, len_n, i, npu_sha)
            diff = int(np.count_nonzero(np.frombuffer(ort_b, np.uint8) != np.frombuffer(npu_b, np.uint8)))
            blob = labelmap.encode(ort_b, W, H, i)
            off = blobs.tell()
            blobs.write(blob)
            blobs.flush()
            row = {'item_index': i, 'source_item_id': rec['source_item_id'], 'input_sha256': expected[i],
                   'ort_labels_sha256': sha(ort_b), 'npu_labels_sha256': npu[i][3], 'rle_offset': off,
                   'rle_length': len(blob), 'different_pixels': diff, 'agreement': 1 - diff / (W * H), 'ort_ms': ms}
            items.write(json.dumps(row) + '\n')
            items.flush()
            print(f'ITEM {i} agreement={row["agreement"]:.5f} ort_ms={ms:.0f}', flush=True)
    return 0


def score(a, lines, index_bytes, rng) -> int:
    from sdv_bmt.metrics.quality_segmentation import ConfusionCounts, ground_truth
    rows = {}
    for x in (a.out / 'items.jsonl').read_text().splitlines():
        r = json.loads(x)
        if r['item_index'] in rows:
            raise SystemExit(f'item {r["item_index"]} twice')
        rows[r['item_index']] = r
    if sorted(rows) != list(rng):
        raise SystemExit(f'coverage: {len(rows)} items, expected {len(rng)}')
    env = json.loads((a.out / 'infer_env.json').read_text())
    if env['index_sha256'] != sha(index_bytes):
        raise SystemExit('index digest differs from the infer step')
    total = ConfusionCounts()
    for i in rng:
        r = rows[i]
        ort_b = read_blob(a.out / 'ort_labels.rle', r['rle_offset'], r['rle_length'], i, r['ort_labels_sha256'])
        gw, gh, gt = ground_truth(json.loads(lines[i])['annotation'])
        total.add(ort_b, gt)
    miou, ious = total.result()
    agree = sorted(r['agreement'] for r in rows.values())
    diff_total = sum(r['different_pixels'] for r in rows.values())
    ms = sorted(r['ort_ms'] for r in rows.values())
    report = {'status': 'COMPLETED', 'items': len(rows), 'infer_env': env, 'score_python': platform.python_version(),
              'scope': 'L1 offline: ORT FP32 CPU vs NPU labels on byte-identical canonical inputs; not an SDV-BMT run',
              'metric': 'sdv_bmt ConfusionCounts (unmodified), summed TP/FP/FN over all items',
              'label_agreement_npu_vs_ort': {
                  'overall': 1 - diff_total / (len(rows) * W * H), 'min': agree[0], 'p50': agree[len(agree) // 2],
                  'max': agree[-1], 'different_pixels_total': diff_total,
                  'items_identical': sum(r['different_pixels'] == 0 for r in rows.values()),
                  'worst_items': sorted((r['agreement'], r['item_index'], r['source_item_id']) for r in rows.values())[:5]},
              'ort_fp32_miou': miou, 'npu_miou_l1': NPU_MIOU_L1, 'gpu_fp32_miou_reference': GPU_FP32_MIOU,
              'npu_minus_ort_miou': NPU_MIOU_L1 - miou,
              'ort_per_class_iou': {CLASSES[c]: v for c, v in ious.items()},
              'ort_counts': {'tp': total.tp, 'fp': total.fp, 'fn': total.fn, 'valid_pixels': total.pixels},
              'ort_ms_cpu': {'min': ms[0], 'p50': ms[len(ms) // 2], 'max': ms[-1]},
              'g15': 'criterion [DECISION PENDING]: proposed drop <= 0.01 for BF16'}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('DONE ' + json.dumps({k: report[k] for k in ('items', 'ort_fp32_miou', 'npu_miou_l1', 'npu_minus_ort_miou',
                                                       'label_agreement_npu_vs_ort')}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--range', default='0:500', help='START:COUNT over the val500 index')
    ap.add_argument('--index', type=Path, default=ROOT / 'datasets/cityscapes-val500/index.jsonl')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input-digests', type=Path, metavar='OUT.json')
    mode.add_argument('--infer', action='store_true')
    mode.add_argument('--score', action='store_true')
    ap.add_argument('--expect-input-digests', type=Path)
    ap.add_argument('--npu', type=Path, nargs='+', help='l1_native output dirs (items.jsonl + labels.rle)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    start, count = (int(x) for x in a.range.split(':'))
    index_bytes = a.index.read_bytes()
    lines = index_bytes.decode().splitlines()
    if not (0 <= start and count > 0 and start + count <= len(lines)):
        raise SystemExit('range outside index')
    rng = range(start, start + count)
    if a.input_digests:
        d = {'python': platform.python_version(), 'index_sha256': sha(index_bytes),
             'items': {str(i): sha(canonical(a.index, json.loads(lines[i]), i)) for i in rng}}
        a.input_digests.parent.mkdir(parents=True, exist_ok=True)
        a.input_digests.write_text(json.dumps(d, indent=1) + '\n')
        print(f'wrote {len(d["items"])} input digests (Python {d["python"]})')
        return 0
    if not a.out:
        raise SystemExit('--out is required')
    if a.score:
        if sys.version_info < (3, 10):
            raise SystemExit('--score must run with the project Python (>= 3.10)')
        return score(a, lines, index_bytes, rng)
    a.out.mkdir(parents=True, exist_ok=True)
    return infer(a, lines, index_bytes, rng)


if __name__ == '__main__':
    raise SystemExit(main())
