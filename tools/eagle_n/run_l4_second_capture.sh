#!/usr/bin/env bash
# L4 (guide Step 6.3): second capture with the SAME provisional plan, golden extracted with the original
# tools/stage2b_g_real/capture_golden.py, then compared row by row with the first golden (identity rate).
# Run as bmt after `sudo -v`, NPU idle. ~20 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
EVID=/home/bmt/caontd/eagle_n_integration_evidence
GOLD=/home/bmt/caontd/eagle_n_integration_assets/r18/npu-capture.golden.jsonl
test "$(sha256sum "$GOLD" | cut -d' ' -f1)" = af30e654681629e2c94044abc094be3059390a090131abc01486530a70012a9a
tools/eagle_n/run_l3_capture_provisional.sh
RUN=$(ls -td "$EVID"/run_* | head -1)
BUNDLE=$(ls -d "$RUN"/bundles/*/run-* | head -1)
SECOND="$RUN/second_capture.golden.jsonl"
.venv/bin/python -B tools/stage2b_g_real/capture_golden.py "$BUNDLE" "$SECOND"
python3 - "$GOLD" "$SECOND" <<'PY' | tee "$RUN/identity.json"
import json, sys
a = {json.loads(l)["item_index"]: json.loads(l) for l in open(sys.argv[1])}
b = {json.loads(l)["item_index"]: json.loads(l) for l in open(sys.argv[2])}
same = sum(a[i] == b.get(i) for i in a)
print(json.dumps({"golden": sys.argv[1], "second": sys.argv[2], "items": len(a), "identical": same,
                  "identity_rate": same / len(a), "differing": [i for i in a if a[i] != b.get(i)][:20]}, indent=1))
PY
