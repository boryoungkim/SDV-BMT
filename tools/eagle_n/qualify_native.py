#!/usr/bin/env python3
"""Write the Eagle-N native qualification receipt from existing, hash-bound L1 evidence. No device access.

ORIGIN OF THIS GATE: internal (Cao-owned paths). `native_qualification.json` is a local launcher/backend guard
(tools/eagle_n/in_container.py, native/src/eagle_n/backend.hpp). It is NOT a SDV-BMT or porting-guide requirement;
the guide's acceptance is L0-L5. The guard enforces guide rules locally: an executor may run non-trial benchmark
plans only after the same executor/plan/weights passed the offline numeric check on the NPU (guide L1) in the same
trace-replay mode the backend uses.

Never written by hand. The receipt is produced only if ALL hold:
  - current executor/plan/weights files hash to the values pinned in the DUT/workload configs;
  - L1 NPU merged report: COMPLETED, every item 0..N-1, same executor/plan/weights/index digests as now, every
    golden check equal to the validated labels; every part ran non-trial (trace replay) and exited 0;
  - L1 ORT report: COMPLETED over the same N items and index, built from exactly those NPU parts;
  - provisional internal sanity bound: |NPU mIoU - ORT mIoU| <= 0.01 (the PROPOSED G15 BF16 figure; G15 itself is
    a guide criterion still [DECISION PENDING] and is NOT decided by this receipt).
Any rebuild of the executor changes its hash, which makes the receipt stale (L1 must be rerun).
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATED = '621089306df561378d3c9aa56901184eda76c79ac46a156e7deaf6ce07935168'
# R50 (INTERNAL anchor, 2026-11-16): the L0 device labels of the standalone golden image (evidence log §48,
# l1_20261116T143507_881070Z), agreement 0.999115 with the ORT FP32 labels. R50 L1 golden checks are rows with
# agreement_vs_ort_fp32 >= min_agreement (tools/eagle_n/l1_native.py GOLDENS); here they must also equal this hash.
R50_L0_LABELS = '3986cee539de634b761fa9cc84564717678f665e7cb44a974bc4e963e4758dcc'
R101_L0_LABELS = '91e23e4100ac47ddecc1ad905d6a5588a167b665c4f1c7752d4bbb34a23d21c6'   # L0 §64, 2026-11-17
NONR18_L0_LABELS = {'r50': R50_L0_LABELS, 'r101': R101_L0_LABELS}
SANITY_BOUND = 0.01


def fsha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit('NOT QUALIFIED: ' + msg)


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dut', type=Path, default=ROOT / 'configs/duts/eagle-n-r18.yaml')
    ap.add_argument('--workload', type=Path, default=ROOT / 'configs/workloads/deeplabv3plus-r18-cityscapes-v1-eagle-n.yaml')
    ap.add_argument('--index', type=Path, default=ROOT / 'datasets/cityscapes-val500/index.jsonl')
    ap.add_argument('--l1-npu', type=Path, required=True, help='l1_merged_* dir (l1_npu_val500.json)')
    ap.add_argument('--l1-ort', type=Path, required=True, help='l1_ort_* dir (report.json)')
    ap.add_argument('--expect', type=int, default=500)
    ap.add_argument('--replace', action='store_true', help='back up and replace an existing receipt')
    a = ap.parse_args()

    opts = yaml.safe_load(a.dut.read_text())['runtime_options']
    prog = yaml.safe_load(a.workload.read_text())['options']['device_program']
    cur = {'executor_sha256': fsha(opts['executor']), 'plan_sha256': fsha(prog['plan']),
           'weights_sha256': fsha(prog['weights']), 'index_sha256': fsha(a.index)}
    need(cur['executor_sha256'] == opts['executor_sha256'], 'executor file differs from the DUT config pin')
    need(cur['plan_sha256'] == prog['plan_sha256'], 'plan file differs from the workload pin')
    need(cur['weights_sha256'] == prog['weights_sha256'], 'weights file differs from the workload pin')

    npu_file = a.l1_npu / 'l1_npu_val500.json'
    npu = json.loads(npu_file.read_text())
    need(npu.get('status') == 'COMPLETED' and npu.get('items') == a.expect, 'L1 NPU report not COMPLETED over all items')
    for k, v in cur.items():
        need(npu.get(k) == v, f'L1 NPU {k} differs from the current file')
    golden = npu.get('golden_checks', [])
    variant = prog['variant']
    need(npu.get('variant', 'r18') == variant, f'L1 NPU variant {npu.get("variant")} differs from workload {variant}')
    if variant == 'r18':
        need(len(golden) >= 2 and all(g['equals_validated'] and g['labels_sha256'] == VALIDATED for g in golden),
             'L1 golden checks missing or not equal to the validated labels')
    elif variant in NONR18_L0_LABELS:   # r50, r101: same rule, own L0 labels
        need(len(golden) >= 2 and all(g['labels_sha256'] == NONR18_L0_LABELS[variant] and g['same_as_previous_golden']
                                      and g['agreement_vs_ort_fp32'] >= g['min_agreement'] >= 0.99 for g in golden),
             f'L1 golden checks missing, not equal to the {variant.upper()} L0 labels or below the ORT agreement floor')
    else:
        need(False, f'no golden rule for variant {variant}')
    evidence = {str(npu_file): fsha(npu_file)}
    covered = []
    for part in map(Path, npu['parts']):
        rep_f = part / 'report.json'
        rep = json.loads(rep_f.read_text())
        need(rep.get('status') == 'COMPLETED', f'{part}: status {rep.get("status")}')
        need(rep.get('mode', '').startswith('non-trial') and 'trace replay' in rep['mode'], f'{part}: not trace-replay mode')
        need(rep['describe'].get('eagle_n.completion', '').startswith('synchronous trace replay'), f'{part}: describe mode')
        for k in ('executor_sha256', 'plan_sha256', 'weights_sha256', 'index_sha256'):
            need(rep.get(k) == cur[k], f'{part}: {k} differs')
        st = json.loads((part.parent / 'status.json').read_text())
        need(st.get('exit_code') == 0 and st.get('npu_reset') is False, f'{part}: launcher status {st}')
        s, n = rep['range']
        covered += range(s, s + n)
        for f in (rep_f, part / 'items.jsonl', part / 'labels.rle', part.parent / 'status.json'):
            evidence[str(f)] = fsha(f)
    need(sorted(covered) == list(range(a.expect)), 'NPU parts do not cover every item exactly once')

    ort_file = a.l1_ort / 'report.json'
    ort = json.loads(ort_file.read_text())
    need(ort.get('status') == 'COMPLETED' and ort.get('items') == a.expect, 'L1 ORT report not COMPLETED over all items')
    env = ort['infer_env']
    need(env['index_sha256'] == cur['index_sha256'] and env['onnx_sha256'] == json.loads(Path(prog['plan']).read_text())['onnx_sha256'],
         'L1 ORT index/ONNX identity differs')
    need(sorted(env['npu_parts']) == sorted(npu['parts']), 'L1 ORT compared different NPU parts')
    need(ort['npu_miou_l1'] == npu['npu_miou'], 'L1 ORT report used a different NPU mIoU')
    for f in (ort_file, a.l1_ort / 'items.jsonl', a.l1_ort / 'ort_labels.rle', a.l1_ort / 'infer_env.json'):
        evidence[str(f)] = fsha(f)
    diff = abs(npu['npu_miou'] - ort['ort_fp32_miou'])
    need(diff <= SANITY_BOUND, f'|NPU - ORT mIoU| = {diff} above the internal sanity bound')

    out = Path(prog['qualification'])
    receipt = {
        'passed': True, 'variant': prog['variant'],
        'executor_sha256': cur['executor_sha256'], 'plan_sha256': cur['plan_sha256'],
        'weights_sha256': cur['weights_sha256'], 'index_sha256': cur['index_sha256'],
        'origin': 'INTERNAL local guard (Cao-owned paths); not a SDV-BMT or porting-guide requirement',
        'basis': 'guide L1 offline numeric check on the NPU, same executor/plan/weights, non-trial trace replay',
        'gate': {'items': a.expect, 'golden_checks_equal_validated': len(golden),
                 'npu_miou': npu['npu_miou'], 'ort_fp32_miou': ort['ort_fp32_miou'], 'abs_miou_diff': diff,
                 'internal_sanity_bound': SANITY_BOUND,
                 'label_agreement_npu_vs_ort': ort['label_agreement_npu_vs_ort']['overall']},
        'not_established': ['G15 acceptance (guide, DECISION PENDING)', 'L3-L5', 'latency / DUT-service',
                            'measured-path allocation audit beyond trace replay'],
        'evidence_sha256': evidence,
        'tool': {'path': str(Path(__file__).resolve()), 'sha256': fsha(Path(__file__)), 'python': sys.version.split()[0]},
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    # The native backend parses this file with the SDV-BMT runner JSON parser, which accepts integers only
    # (native/src/common/json.hpp): every non-integer number is written as a decimal string.
    def ints_only(v):
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, dict):
            return {k: ints_only(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [ints_only(x) for x in v]
        return v
    receipt = ints_only(receipt)
    text = json.dumps(receipt, indent=1) + '\n'
    json.loads(text, parse_float=lambda x: need(False, f'float token {x} in receipt'))
    if out.exists():
        need(a.replace, f'{out} exists; use --replace to back it up and replace it')
        shutil.copy2(out, out.with_name(out.name + '.bak.' + receipt['generated_utc'].replace(':', '')))
    out.write_text(text)
    print(json.dumps({'written': str(out), 'sha256': fsha(out), **{k: receipt[k] for k in
                      ('executor_sha256', 'plan_sha256', 'weights_sha256')}, 'gate': receipt['gate']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
