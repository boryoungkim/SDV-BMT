# LLM benchmark (Llama 3 on GPU and Furiosa RNGD)

This page describes how SDV-BMT measures Llama 3.2 1B / 3B Instruct and Llama 3.1 8B Instruct on an NVIDIA
GPU (vLLM) and on a FuriosaAI RNGD card (furiosa-llm), and how to reproduce the runs.

The LLM path is a **Python measurement plane**, separate from the C++ measurement plane used for the CNN and
LiDAR workloads. Its numbers are evaluated with their own catalog (`sdv-bmt-llm-v1`) and are not comparable
with the Metric Set v1 / v1.1 / v1.2 numbers of the other workloads.

GPU and RNGD results describe whole systems (runtime + driver + device), shown side by side. They are not
attributed to a chip. The reference results are in
[`docs/results/llm-llama3-gpu-rngd-result.md`](results/llm-llama3-gpu-rngd-result.md).

## 1. What is measured

| Item | Definition |
|---|---|
| Request | One generation call with pre-tokenized input token IDs, one at a time (batch 1, concurrency 1) |
| Clock | Host `time.monotonic_ns` in the Python process that drives the runtime's streaming API |
| t_submit | Just before the streaming call is made |
| t_first | Receipt of the first stream item that carries at least one output token ID |
| t_done | Receipt of the stream item that carries the finish reason |
| Tokenization | Input tokenization and chat template are applied before the run and are not timed. Output text is decoded after t_done. Runtime-internal detokenization is turned off where the runtime allows it (vLLM `detokenize=False`; furiosa-llm native token-ID stream) |
| Not observed | Device-internal time |

With N = number of output token IDs and n_first = token IDs in the first stream item:

- TTFT = t_first − t_submit
- E2E = t_done − t_submit
- TPOT = (t_done − t_first) / (N − n_first)
- decode throughput = (N − n_first) / (t_done − t_first)

When the runtime streams one token per item, n_first = 1. If it delivers several tokens in one item
("chunk"), only item arrival times are observable. The evaluation then reports the stream unit as `chunk`
and per-token arrival as not observable.

Statistics are reported as raw values, median (nearest-rank p50), min, max, p90 and max_observed. p99 is
reported only with at least 100 samples. Every percentile carries its sample count. max_observed is not a
worst-case execution time.

### Controls

Every control is recorded as the value the runtime reports after loading:

- greedy decoding (temperature 0) on every request;
- prefix caching off. Confirmed by the applied setting, the per-request `num_cached_tokens` and the runtime's
  prefix-cache hit counters. Timing is used only as a diagnostic;
- no speculative decoding;
- one request at a time (checked from the request time spans).

The normal in-request KV cache stays on. A control that cannot be read back is reported as `unverified`.

## 2. Workloads

| Workload | Content |
|---|---|
| Speed S1–S5 | Fixed-length synthetic inputs (input/output tokens): S1 128/64, S2 128/256, S3 512/512, S4 830/256, S5 1024/128. BOS + regular token IDs from a SHA-256 counter generator (seed 20261117). EOS ignored, min = max = target tokens. Per S: 3 warm-up + 5 measured requests. File: `configs/llm/speed-s1-s5.tokens.jsonl` |
| Quality | **MMLU 1,000 items, zero-shot generation-based evaluation.** `cais/mmlu` `all` test at revision `c30699e8356da336a370243923dbaf21066bb9fe`, subject-proportional 1,000 items (seed 42), one user message with the model's chat template (date fixed to 26 Jul 2024), greedy, at most 16 new tokens, natural stop at the Llama 3 end tokens. Item list (IDs + SHA-256, no text): `configs/llm/mmlu-1000.items.json` |

A speed request counts as a fixed-length measurement only if it produced exactly the target number of tokens
with finish reason `length`. Any other completed request is counted as a length mismatch and kept out of the
statistics.

MMLU answer extraction applies these rules in order:

1. If the first non-space character is A–D and is not followed by a letter or digit, it is the answer.
2. Otherwise, an "answer is X" phrase gives the answer.
3. Otherwise, the output is unparsable.

Unparsable outputs count as incorrect and are reported separately. Items that did not run make the
evaluation incomplete.

The GPU–RNGD quality criterion is |accuracy difference| ≤ 0.01 on this fixed item set. It is a total-quality
criterion only, not a statement of numerical equivalence or per-item identity.

## 3. Models and artifacts

| Model | Weights (GPU) | RNGD artifact |
|---|---|---|
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` (BF16) | built locally with `tools/llm/rngd_build.py` from the local copy of this revision |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95` (BF16) | built locally with `tools/llm/rngd_build.py` |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` (BF16) | `furiosa-ai/Llama-3.1-8B-Instruct` @ `b678d58b0a13c9a4e94e9d235d70b53fe0d8bf32` (tag `v2026.3`, BF16 weights / activations / KV cache) |

The meta-llama repositories are gated. Each user requests access and accepts the Llama 3.2 / 3.1 Community
License with their own account.

Locally built RNGD artifacts use the SDK's own Llama-3.1-8B bucket preset, restricted to batch 1 and a
2,048-token context: prefill 128–1024 (step 128), decode (1, 1024) and (1, 2048), and the matching
chunked-prefill (append) buckets. Tensor parallelism is 8 PEs (one card). These artifacts are derived
works and are not distributed.

## 4. Environments

| Device | Environment |
|---|---|
| GPU | Python 3.10 venv with vLLM 0.31.0 (torch 2.13.0 CUDA 13.0 build, transformers 5.17.0), pyarrow 25.0.1 and PyYAML 6.0.3, from PyPI |
| RNGD | furiosa-llm 2026.3.0 (furiosa-llm-native 2026.3.0, torch 2.10.0, transformers 5.1.0), installed per the FuriosaAI SDK documentation; firmware 2026.4.0 (recorded in every run), driver 2026.4.0 on the reference host |

Three package lists describe these environments. They have different roles:

| File | Role | Captured |
|---|---|---|
| `tools/llm/requirements-vllm.txt` | Install file for a new GPU environment: `pip install -r tools/llm/requirements-vllm.txt`. It pins only the five key packages and lets pip resolve the rest, so secondary packages can differ from the reference environment | written from the reference environment |
| `tools/llm/requirements-vllm.lock.txt` | Full package list of the reference GPU environment (203 packages, `uv pip freeze`). Use it to compare an environment with the reference or to pin the same versions (installing from this list was not tested) | 2026-11-17 09:14 KST, during the GPU baseline runs |
| `tools/llm/requirements-furiosa-llm.lock.txt` | Full package list of the reference RNGD environment (161 packages, `pip freeze --all`). A reference list: install the FuriosaAI packages per the SDK documentation | 2026-11-17 20:43 KST, after the runs. Nothing was installed into this environment during the work, but the list was not recorded at run time |

What was actually installed and checked:

- **Reference GPU environment:** created with `uv pip install vllm==0.31.0 pyarrow`, constrained to the versions of
  an existing local vLLM environment. That constraints file is not published.
  - `requirements-vllm.txt` was written afterwards. Neither it nor `requirements-vllm.lock.txt` has been used to
    create an environment.
  - The short file therefore does not reproduce every version of the reference environment: pip may resolve
    secondary packages differently. For the exact versions, compare with or pin from the full list.
- **Reference RNGD environment:** installed before this work, per the FuriosaAI SDK documentation. This work
  installed nothing into it.
- **Recorded in the evidence bundles:** only the runtime version (and the torch version on the GPU). The full
  package lists are not part of the bundles.
- **The only installation tested from the published files** is the core's public CI setup:
  `pip install -e ".[dev]"` in a fresh CPU environment.

SDV-BMT itself is not installed into these environments. Run it with `PYTHONPATH=src`; it needs only PyYAML
there.

## 5. Reproduce

```bash
export SDV_BMT_MODELS=/path/to/models          # <name>/ HF snapshots, rngd/<name>-fl2026.3/ artifacts
export SDV_BMT_DATASETS=/path/to/datasets      # mmlu-1000/<name>.prompts.jsonl (generated below)
export SDV_BMT_CACHE=/path/to/cache
export VLLM_PY=/path/to/vllm-venv/bin/python
export FURIOSA_PY=/path/to/furiosa-llm-venv/bin/python
export PYTHONPATH=src

# 0) Assets, with the Hugging Face CLI `hf` (installed in the vLLM environment). The meta-llama repositories
#    are gated: accept their license with your own account first.
for m in "llama-3.2-1b-instruct meta-llama/Llama-3.2-1B-Instruct 9213176726f574b556790deb65791e0c5aa438b6" \
         "llama-3.2-3b-instruct meta-llama/Llama-3.2-3B-Instruct 0cb88a4f764b7a12671c53f0838cd831a0843b95" \
         "llama-3.1-8b-instruct meta-llama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659"; do
  set -- $m; hf download $2 --revision $3 --exclude "original/*" --local-dir $SDV_BMT_MODELS/$1
done
HF_HOME=<HF cache holding the furiosa-ai 8B artifact> hf download furiosa-ai/Llama-3.1-8B-Instruct \
    --revision b678d58b0a13c9a4e94e9d235d70b53fe0d8bf32
hf download cais/mmlu all/test-00000-of-00001.parquet --repo-type dataset \
    --revision c30699e8356da336a370243923dbaf21066bb9fe --local-dir <mmlu>
export HF_HUB_OFFLINE=1                        # everything below runs offline

# 1) MMLU prompts (needs the pinned all/test parquet; verifies the public item list)
$VLLM_PY -m sdv_bmt.llm prepare-mmlu --parquet <mmlu>/all/test-00000-of-00001.parquet \
    --tokenizer $SDV_BMT_MODELS/llama-3.1-8b-instruct --items configs/llm/mmlu-1000.items.json \
    --prompts-out $SDV_BMT_DATASETS/mmlu-1000/llama-3.1-8b-instruct.prompts.jsonl \
    --revision c30699e8356da336a370243923dbaf21066bb9fe
# (repeat for the 1B and 3B tokenizers; all three give the same prompt file,
#  SHA-256 b0129415cc1b683a936b07fcdf92e1c64ecb13b5f1d2e8cc4ca6c01e7ea907c3)
$VLLM_PY -m sdv_bmt.llm lengths $SDV_BMT_DATASETS/mmlu-1000/llama-3.1-8b-instruct.prompts.jsonl --max-model-len 2048

# 2) RNGD artifacts for 1B / 3B (CPU build, about 12 / 19 min on the reference host)
TMPDIR=<short path> $FURIOSA_PY tools/llm/rngd_build.py --model $SDV_BMT_MODELS/llama-3.2-1b-instruct \
    --out $SDV_BMT_MODELS/rngd/llama-3.2-1b-instruct-fl2026.3 --cache-dir $SDV_BMT_CACHE/furiosa-llm-build \
    --num-pipeline-builder-workers 4 --num-compile-workers 8

# 3) Baselines: per model 5 speed runs + 1 MMLU run, one device at a time
HF_HOME=<HF cache holding the furiosa-ai 8B artifact> tools/llm/run_baselines.sh gpu
HF_HOME=<HF cache holding the furiosa-ai 8B artifact> tools/llm/run_baselines.sh rngd

# 4) Summaries
python3 -m sdv_bmt.llm aggregate runs/llm-llama/<experiment>/run-* --out <summary.json>
python3 -m sdv_bmt.llm compare <GPU MMLU run> <RNGD MMLU run> --out <compare.json>
```

`run_baselines.sh` writes `runs/TIMED_RUN_IN_PROGRESS` while it runs, and checks that the device is idle
before every run. GPU runs use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`.

### Supporting checks

| Tool | Purpose |
|---|---|
| `tools/llm/rngd_smoke.py` | furiosa-llm runtime, token-ID stream, fixed length, prefix-cache control (with an `on` positive control), long inputs |
| `tools/llm/vllm_prefix_control.py` | vLLM prefix-cache positive control (off vs on) |
| `tools/llm/hf_check.py` | vLLM BF16 vs Hugging Face transformers FP32 (CPU): first token, 32-token greedy generation, MMLU answer path on 100 items |
| `tools/llm/check_tokenizers.py` | the two tokenizers give token IDs the same meaning (vocabulary, special tokens, chat template, re-encoding of all MMLU prompts) |
| `tools/llm/trace_weights.py` | per-tensor SHA-256 comparison of an RNGD artifact's parameters with the pinned Hugging Face weights |

Each tool writes a result file, but it does not record its own version or the package versions it ran with, and
the evidence bundles do not reference these results. The reference results page labels which statements come from
the bundles and which come from these separate checks.

## 6. Bundle layout

```
runs/llm-llama/<experiment_id>/experiment.json
runs/llm-llama/<experiment_id>/<run_id>/
    run_spec.json  environment.json  requests.jsonl  [quality_reference.jsonl]  manifest.json
    evaluations/<eval_id>/metrics.json  eval_manifest.json
```

Every file is created exclusively, and `manifest.json` is written last with the SHA-256 of every raw file.
`python -m sdv_bmt.llm evaluate RUN_DIR` reads only the bundle. It refuses a bundle whose raw files changed
and writes a new evaluation directory.

Bundles of MMLU runs contain answer letters and outputs derived from the dataset. They stay local, like the
COCO, Cityscapes and nuScenes evidence.

## 7. Limits

- Host-observed times of the whole system. Device-internal times are not observed.
- vLLM may deliver several tokens in one stream item when the Python consumer falls behind. This is reported
  per run.
- RNGD prefill runs in 128-token buckets (up to 1,024 tokens). Longer inputs use chunked prefill. The
  locally built 1B/3B artifacts accept at most 2,048 tokens of input plus output.
- MMLU here is zero-shot and generation-based with a 16-token limit. Outputs that start with an
  explanation can be unparsable, so the scores are not comparable with published 5-shot log-likelihood
  MMLU scores.
