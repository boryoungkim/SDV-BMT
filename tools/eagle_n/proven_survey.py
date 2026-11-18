#!/usr/bin/env python3
"""Proven-shape survey (host only, no device): every SDK call the executor will make for a candidate plan must be
dominated by a call that ALREADY COMPLETED on this NPU.

Lesson of 2026-11-16 (eagle_n_integration_review/BAI_HOC_NPU_TREO_20261116.md): the L1 preflight cannot see device-side
hangs (op #101, 1x1 1024->2048 at 128x256, hung with plenty of L1). Policy: run only call shapes that are proven.

Inputs (all produced by the executor preflight under ttnn NO_DISPATCH, which records every SDK call with its arguments):
  --candidate   preflight.json of the plan to check
  --proven      preflight.json[:LAST_OP_INDEX] of a graph that completed on the device (repeatable); calls of plan ops with
                index > LAST_OP_INDEX are not proven (e.g. the bf16 run that hung at op #101 proves ops #0-#100 only)
  --standalone  standalone R50 V10.1 run folder (operations.jsonl + operator_completions.jsonl): completed ttnn.conv2d calls
A call is proven when a proven call has the same SDK function name, the same number of tensor arguments and every tensor
shape is element-wise <= (right-aligned). conv2d calls may also be proven by a completed standalone conv2d call with the same
kernel and group type and input pixels, input channels and output channels all <= (standalone ran FP32, so bytes >= BF16).
Exit 0 only if every candidate call is proven.
"""
from __future__ import annotations
import argparse, collections, json, re, sys
from pathlib import Path

SHAPE = re.compile(r'(?:Shape\(|(?:out|in):)\[([0-9,\s]*)\]')
CONV = re.compile(r'conv2d', re.I)
# Only device operations create programs that run on the cores: the trace names them by class (Matmul, Conv2d, Tilize,
# SliceDeviceOperation, ...). Wrapper/metadata functions (ttnn::*, Tensor::*, tt::tt_metal::create_device_tensor) are skipped.
DEVICE_OP = re.compile(r'^[A-Z][A-Za-z0-9]*$')

def shapes(args):
    out = []
    for a in args:
        for m in SHAPE.finditer(a):
            dims = [int(x) for x in m.group(1).replace(' ', '').split(',') if x]
            out.append(tuple(dims))
    return out

def dominated(c, p):
    if len(c) != len(p):
        return False
    for a, b in zip(c, p):
        n = max(len(a), len(b)); a = (1,) * (n - len(a)) + a; b = (1,) * (n - len(b)) + b
        if any(x > y for x, y in zip(a, b)):
            return False
    return True

def load_calls(path, last=None):
    r = json.loads(Path(path).read_text())
    calls = []
    for op in r['ops']:
        if last is not None and op['index'] > last:
            continue
        for c in op.get('calls', []):
            if DEVICE_OP.match(c['n']) and shapes(c['a']):
                calls.append((c['n'], tuple(shapes(c['a']))))
    return calls, r

def conv_sig(shs):
    """(in_pixels, in_ch, out_ch, kh, kw, depthwise) from conv2d tensor args: input [..,H*W,C] or [1,H,W,C]; weight [N,Cg,kh,kw]."""
    four = [s for s in shs if len(s) == 4]
    if len(four) < 2:
        return None
    x, w = four[0], four[1]
    in_px = x[1] * x[2] if x[1] > 1 else x[2]
    n, cg, kh, kw = w
    return in_px, x[3], n, kh, kw, cg == 1 and n > 1

def standalone_sigs(folder):
    folder = Path(folder)
    done = {json.loads(l)['name'] for l in open(folder / 'operator_completions.jsonl') if 'device_completion_observed' in l}
    sigs = []
    for line in open(folder / 'operations.jsonl'):
        r = json.loads(line)
        if r['operator'] != 'ttnn.conv2d' or r['name'] not in done:
            continue
        _, c, h, w = r['input']; kh, kw = r['kernel']; g = r['groups']
        sigs.append((h * w, c, r['out_channels'], kh, kw, g > 1 and g == c))
    return sigs

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--proven', action='append', default=[])
    ap.add_argument('--standalone')
    ap.add_argument('--out', required=True)
    ap.add_argument('--show', type=int, default=40)
    a = ap.parse_args()
    proven = collections.defaultdict(list)
    sources = []
    for spec in a.proven:
        path, _, last = spec.partition(':')
        calls, _ = load_calls(path, int(last) if last else None)
        for n, s in calls:
            proven[n].append(s)
        sources.append({'preflight': path, 'last_op_index': int(last) if last else None, 'calls': len(calls)})
    sa = standalone_sigs(a.standalone) if a.standalone else []
    sources.append({'standalone': a.standalone, 'conv2d_calls': len(sa)})
    for n in proven:   # dedupe, keep maximal shapes first (speed)
        proven[n] = sorted(set(proven[n]), key=lambda s: -sum(sum(t) for t in s))
    cand = json.loads(Path(a.candidate).read_text())
    report = {'schema': 'eagle-n-proven-survey/1', 'candidate': a.candidate, 'sources': sources, 'ops': []}
    bad_total = calls_total = 0
    unseen_names = collections.Counter()
    for op in cand['ops']:
        bad = []
        for c in op.get('calls', []):
            n, shs = c['n'], tuple(shapes(c['a']))
            if not DEVICE_OP.match(n):
                continue
            calls_total += 1
            if not shs:   # never let an unparsed call pass trivially (lesson: 'unsupported type' arguments, 2026-11-16)
                bad.append({'name': n, 'shapes': [], 'reason': 'no tensor shape recorded'})
                continue
            ok = any(dominated(shs, p) for p in proven.get(n, ()))
            via = 'native' if ok else None
            if not ok and CONV.search(n) and sa:
                sig = conv_sig(shs)
                if sig and any(s[3:] == sig[3:] and all(x <= y for x, y in zip(sig[:3], s[:3])) for s in sa):
                    ok, via = True, 'standalone'
            if not ok:
                bad.append({'name': n, 'shapes': [list(s) for s in shs]})
                if n not in proven:
                    unseen_names[n] += 1
        bad_total += len(bad)
        report['ops'].append({'index': op['index'], 'kind': op['kind'], 'output': op['output'],
                              'launch': op.get('launch'), 'calls': sum(1 for c in op.get('calls', []) if DEVICE_OP.match(c['n'])), 'unproven': bad})
    report.update(calls_checked=calls_total, unproven_calls=bad_total,
                  unproven_ops=sum(1 for o in report['ops'] if o['unproven']),
                  function_names_never_proven=dict(unseen_names), passed=bad_total == 0 and calls_total > 0)
    Path(a.out).write_text(json.dumps(report, indent=1) + '\n')
    print(f"SURVEY calls checked {calls_total}, unproven calls {bad_total} in {report['unproven_ops']} plan ops; "
          f"passed={report['passed']}")
    shown = 0
    for o in report['ops']:
        if o['unproven'] and shown < a.show:
            first = o['unproven'][0]
            print(f"  #{o['index']:<3} {o['kind']:7s} {(o['launch'] or {}).get('config', '-'):18s} {o['output'][-48:]:48s} "
                  f"unproven {len(o['unproven'])}/{o['calls']} e.g. {first['name']} {first['shapes']}")
            shown += 1
    if unseen_names:
        print('  SDK functions never seen in any proven run:', dict(unseen_names))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
