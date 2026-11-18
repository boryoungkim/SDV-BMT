#!/usr/bin/env python3
"""Merge L1 partial runs (tools/eagle_n/l1_native.py) into one exact val500 result. No device access.

mIoU is computed from the SUMMED per-class TP/FP/FN of all items (Cityscapes definition), never as an
average of partial mIoUs. Requires every item 0..N-1 exactly once, identical executor/plan/weights/index
digests in all parts, and every golden check passing its variant anchor (l1_native.py GOLDENS):
r18 = equal to the validated labels; r50 = agreement vs ORT FP32 >= floor and one identical hash in all parts.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
# docs/results/v0.1.2-gpu-reference-result.md, 500 val images
GPU_FP32_MIOU = {'r18': 0.767566, 'r50': 0.798469, 'r101': 0.808709}
G15 = {'bf16': 'proposed drop <= 0.01 for BF16', 'bfp8w': 'proposed drop <= 0.02 for an 8-bit format (bfloat8_b)'}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('parts', nargs='+', type=Path, help='l1 output dirs (with report.json, items.jsonl)')
    ap.add_argument('--expect', type=int, default=500)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    keys = ('executor_sha256', 'plan_sha256', 'weights_sha256', 'index_sha256')
    ident, seen, tp, fp, fn, times, golden = None, {}, [0] * 19, [0] * 19, [0] * 19, [], []
    pixels = 0
    for p in a.parts:
        rep = json.loads((p / 'report.json').read_text())
        if rep.get('status') != 'COMPLETED':
            raise SystemExit(f'{p}: status {rep.get("status")}')
        this = {k: rep[k] for k in keys}
        if ident is None:
            ident = this
        elif this != ident:
            raise SystemExit(f'{p}: digests differ from the first part: {this} vs {ident}')
        golden += rep['golden_checks']
        variant, precision = rep.get('variant', 'r18'), rep.get('plan_precision', 'bf16')   # pre-2026-11-16 reports: R18
        for line in (p / 'items.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['item_index'] in seen:
                raise SystemExit(f'item {r["item_index"]} appears twice')
            seen[r['item_index']] = r['labels_sha256']
            for c in range(19):
                tp[c] += r['tp'][c]; fp[c] += r['fp'][c]; fn[c] += r['fn'][c]
            pixels += r['valid_pixels']
            times.append(r['run_ms'])
    if sorted(seen) != list(range(a.expect)):
        raise SystemExit(f'coverage: {len(seen)} items, missing {sorted(set(range(a.expect)) - set(seen))[:10]}')
    if variant == 'r18':
        if not all(g['equals_validated'] for g in golden):
            raise SystemExit('a golden check differs from the validated labels')
    elif not (all(g['agreement_vs_ort_fp32'] >= g['min_agreement'] for g in golden)
              and len({g['labels_sha256'] for g in golden}) == 1):
        raise SystemExit('a golden check is below the ORT agreement floor or not identical across parts')
    ious = {}
    for c in range(19):
        u = tp[c] + fp[c] + fn[c]
        ious[CLASSES[c]] = tp[c] / u if u else None
    present = [v for v in ious.values() if v is not None]
    miou = sum(present) / len(present)
    s = sorted(times)
    result = {'status': 'COMPLETED', 'items': len(seen), 'parts': [str(p) for p in a.parts], **ident,
              'variant': variant, 'plan_precision': precision,
              'npu_miou': miou, 'gpu_fp32_miou_reference': GPU_FP32_MIOU[variant],
              'miou_drop_vs_gpu': GPU_FP32_MIOU[variant] - miou,
              'per_class_iou': ious, 'valid_pixels': pixels, 'golden_checks': golden,
              'run_ms_host_observed': {'n': len(s), 'min': s[0], 'p50': s[len(s) // 2], 'p99': s[int(len(s) * 0.99)],
                                       'max': s[-1], 'note': 'diagnostic host time around run(); not DUT-service'},
              'not_yet': 'ORT FP32 per-image label agreement on the same canonical tensors (guide L1 item 2/4)',
              'g15': 'criterion [DECISION PENDING]: ' + G15.get(precision, 'see guide')}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'l1_npu_val500.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('items', 'npu_miou', 'gpu_fp32_miou_reference', 'miou_drop_vs_gpu',
                                             'run_ms_host_observed')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
