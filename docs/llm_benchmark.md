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
| Quality | **tinyMMLU 100 items, evaluated as the lm-evaluation-harness task `tinyMMLU`.** `tinyBenchmarks/tinyMMLU` `all` test (all 100 items, dataset order) at revision `cd6a8ee9e1e6f3f18f683535f6c096c386a85da3`. Prompt = the item's `input_formatted` text unchanged (the Open LLM Leaderboard MMLU prompt with its five solved examples already inside), no chat template. Item list (IDs + SHA-256, no text): `configs/llm/tinymmlu-100.items.json` |

A speed request counts as a fixed-length measurement only if it produced exactly the target number of tokens
with finish reason `length`. Any other completed request is counted as a length mismatch and kept out of the
statistics.

tinyMMLU scoring follows the harness (`output_type: multiple_choice`, lm-evaluation-harness 0.4.13):

1. For each choice `" A"` … `" D"` the request is the context + continuation token IDs, built as the harness
   builds them for a causal model: `input_formatted + " " + letter` is encoded with the tokenizer's default special
   tokens (one BOS), and the continuation is the part after the separately encoded context.
2. Each request generates one token and asks the runtime for the log-probability of every prompt token
   (`prompt_logprobs`). The choice's log-likelihood is the sum over the continuation tokens (one token here).
3. The prediction is the first choice with the largest log-likelihood (as numpy `argmax`). The harness metric
   `acc_norm` divides by the choice length in characters; the four lengths are equal, so it equals `acc`.
4. The evaluation reports the raw accuracy over the 100 items (denominator always 100) and the per-item correctness
   vector in dataset order. The value the harness reports, the tinyBenchmarks **gp-IRT** estimate of full-MMLU
   accuracy, is computed from that vector by `tools/llm/tinymmlu_irt.py` with a pinned IRT parameter file.

An item is scored only when all four requests completed with log-probabilities; otherwise it is `failed` and the
evaluation is incomplete.

Equivalence with the harness was checked on the GPU for the three models with `tools/llm/tinymmlu_harness_check.py`
(§5). The harness reads the same 100 documents and builds the same token IDs for all 400 requests. Predictions
differ on 0–3 items per model and setting (Llama 3.2 1B under batch 1 and prefix caching off: identical predictions
and the identical reported value). Every differing item is a tie on one side (equal top-two log-likelihoods, broken
toward the first choice) and a one-step BF16 margin (0.125) on the other. The harness itself moves by one item when
only its batching and prefix-caching settings change. On 100 items, expect 1–3 items (about 0.006 in gp-IRT)
between implementations or execution settings.

#### GPU–RNGD comparison: the near-tie rule

`python -m sdv_bmt.llm compare <GPU run> <RNGD run>` checks whether the two systems compute the same answers.

Why a plain score difference is not enough: each choice gets a score, and the highest score is the answer. In BF16
the scores move in steps (0.0625–0.125 for these models), and two systems can land one step apart. When the best two
choices are that close, one step decides the answer. Example (Llama 3.2 1B, item `test/16`):

| System | A | B | gap between the best two | answer |
|---|---|---|---|---|
| GPU | −1.0327 | −0.9077 | 0.125 | B |
| RNGD | −0.9839 | −0.9839 | 0 (tie: the first choice wins) | A |

Such answer changes say nothing about whether the computation is right; the same GPU changes them when only the
harness's batching settings change.

The rule:

1. An item is **near-tie** when, on either system, the best and second-best scores are at most 0.25 apart (two
   steps). Every other scored item is **clear**.
2. **Pass** when the two systems give the same answer on every clear item; **fail** when one clear answer differs.
3. Near-tie items are counted and listed with both systems' scores, not judged.

The result also keeps the plain numbers (accuracy difference, items answered differently, largest score difference)
and a one-line summary, for example `pass: on the 81 items where both sides rank the best choice clearly
(gap > 0.25), 0 answers differ; 4 of the 19 near-tie items differ`. The 0.25 gap fits the step size of these BF16
models; check it again for a model whose scores move in larger steps, or for a system that computes in lower
precision (INT8 / FP8), where clear answers can change on purpose and a quality tolerance is the right question.

Reference results (commit runs, 2026-11-25): Llama 3.2 1B / 3B and Llama 3.1 8B pass, with 71 / 79 / 81 clear items
and 0 differing clear answers; 2 / 3 / 4 of the 29 / 21 / 19 near-tie items differ.

The v1.0 quality workload, MMLU 1,000 items with a generation-based evaluation (`cais/mmlu`, subject-proportional
selection, chat template, answer-letter extraction), was replaced by tinyMMLU. Its code (`sdv_bmt.llm.mmlu`,
`prepare-mmlu`) and item list (`configs/llm/mmlu-1000.items.json`) remain so that v1.0 bundles can still be
evaluated; its experiment configurations are in the `v1.0` tag.

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

The speed runs use these 2,048-token artifacts. The longest tinyMMLU request is 3,029 tokens, so the tinyMMLU runs
of 1B and 3B use a second artifact built from the same weights with `--max-model-len 4096` (same preset restricted to
4,096 tokens: decode buckets up to 4,096 and 34 append buckets), `rngd/<model>-fl2026.3-4k`. The 8B artifact accepts
131,072 tokens and serves both.

## 4. Environments

| Device | Environment |
|---|---|
| GPU | Python 3.10 venv with vLLM 0.31.0 (torch 2.13.0 CUDA 13.0 build, transformers 5.17.0), pyarrow 25.0.1 and PyYAML 6.0.3, from PyPI |
| RNGD | furiosa-llm 2026.3.0 (furiosa-llm-native 2026.3.0, torch 2.10.0, transformers 5.1.0), installed per the FuriosaAI SDK documentation; firmware 2026.4.0 (recorded in every run), driver 2026.4.0 on the reference host |
| tinyMMLU tools (offline) | a separate venv: `uv pip install -r tools/llm/requirements-vllm.lock.txt lm-eval==0.4.13 "git+https://github.com/felipemaiapolo/tinyBenchmarks@e9a8b1031b0340571beb6c9ca3a27891be09a8fd"`. Used only by `tinymmlu_irt.py` and `tinymmlu_harness_check.py`, never by a run |

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
export SDV_BMT_DATASETS=/path/to/datasets      # tinymmlu-100/<name>.prompts.jsonl (generated below)
export SDV_BMT_CACHE=/path/to/cache
export VLLM_PY=/path/to/vllm-venv/bin/python
export FURIOSA_PY=/path/to/furiosa-llm-venv/bin/python
export LMEVAL_PY=/path/to/lm-eval-venv/bin/python   # tinyMMLU tools only (§4)
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
hf download tinyBenchmarks/tinyMMLU all/test-00000-of-00001.parquet --repo-type dataset \
    --revision cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 --local-dir <tinymmlu>
#    (SHA-256 256f81ffdd6e64699df096c02afafba33626bfca08baed484c275e351993f317)
curl -fL -o <irt>/tinyBenchmarks.pkl https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/e9a8b1031b0340571beb6c9ca3a27891be09a8fd/tinyBenchmarks/tinyBenchmarks.pkl
#    (SHA-256 c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284; the gp-IRT parameters)
export HF_HUB_OFFLINE=1                        # everything below runs offline (except the harness check)

# 1) tinyMMLU requests (verifies the public item list; 4 requests per item)
$VLLM_PY -m sdv_bmt.llm prepare-tinymmlu --parquet <tinymmlu>/all/test-00000-of-00001.parquet \
    --tokenizer $SDV_BMT_MODELS/llama-3.1-8b-instruct --items configs/llm/tinymmlu-100.items.json \
    --prompts-out $SDV_BMT_DATASETS/tinymmlu-100/llama-3.1-8b-instruct.prompts.jsonl \
    --revision cd6a8ee9e1e6f3f18f683535f6c096c386a85da3
# (repeat for the 1B and 3B tokenizers; all three give the same request file,
#  SHA-256 9b3436e1166390acb211a55ded21d97e0e5094b4539d17118f1d7a9c1267b00c, 400 requests, 253-3,029 tokens)
$VLLM_PY -m sdv_bmt.llm lengths $SDV_BMT_DATASETS/tinymmlu-100/llama-3.1-8b-instruct.prompts.jsonl --max-new 1 --max-model-len 4096

# 2) RNGD artifacts for 1B / 3B (CPU build, about 12 / 19 min on the reference host)
TMPDIR=<short path> $FURIOSA_PY tools/llm/rngd_build.py --model $SDV_BMT_MODELS/llama-3.2-1b-instruct \
    --out $SDV_BMT_MODELS/rngd/llama-3.2-1b-instruct-fl2026.3 --cache-dir $SDV_BMT_CACHE/furiosa-llm-build \
    --num-pipeline-builder-workers 4 --num-compile-workers 8
#    and the tinyMMLU artifacts: the same with --max-model-len 4096 --out $SDV_BMT_MODELS/rngd/<model>-fl2026.3-4k
#    (about 4 / 7 min on the reference host)

# 3) Baselines: per model 5 speed runs + 1 tinyMMLU run, one device at a time (REPS=0: tinyMMLU only)
HF_HOME=<HF cache holding the furiosa-ai 8B artifact> tools/llm/run_baselines.sh gpu
HF_HOME=<HF cache holding the furiosa-ai 8B artifact> tools/llm/run_baselines.sh rngd

# 4) Summaries
python3 -m sdv_bmt.llm aggregate runs/llm-llama/<experiment>/run-* --out <summary.json>
python3 -m sdv_bmt.llm compare <GPU tinyMMLU run> <RNGD tinyMMLU run> --out <compare.json>
$LMEVAL_PY tools/llm/tinymmlu_irt.py --pkl <irt>/tinyBenchmarks.pkl --out <irt.json> <tinyMMLU run> [...]

# 5) Metric table (docs/metric_table.md): representative metrics first, then the overall result per metric, then one row per run
python3 -m sdv_bmt.llm table runs/llm-llama/<experiment> [...] --irt <irt.json> --out <metrics.csv>
#    (--irt: the result file of tinymmlu_irt.py; the table then shows the raw tinyMMLU accuracy and the gp-IRT value)
```

`run_baselines.sh` writes `runs/TIMED_RUN_IN_PROGRESS` while it runs, and checks that the device is idle
before every run. GPU runs use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`.

### Supporting checks

| Tool | Purpose |
|---|---|
| `tools/llm/rngd_smoke.py` | furiosa-llm runtime, token-ID stream, fixed length, prefix-cache control (with an `on` positive control), long inputs |
| `tools/llm/vllm_prefix_control.py` | vLLM prefix-cache positive control (off vs on) |
| `tools/llm/tinymmlu_harness_check.py` | tinyMMLU path vs lm-evaluation-harness 0.4.13 itself (vLLM, GPU): same documents, same token IDs for all 400 requests, per-choice log-likelihoods, predictions and reported gp-IRT value; `--sdv-bmt-conditions` runs the harness with batch 1 and prefix caching off |
| `tools/llm/hf_check.py` | vLLM BF16 vs Hugging Face transformers FP32 (CPU): first token, 32-token greedy generation, MMLU answer path on 100 items (v1.0 MMLU workload) |
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

Bundles of tinyMMLU (and v1.0 MMLU) runs contain answer letters and log-likelihoods derived from the dataset. They stay local, like the
COCO, Cityscapes and nuScenes evidence.

## 7. Limits

- Host-observed times of the whole system. Device-internal times are not observed.
- vLLM may deliver several tokens in one stream item when the Python consumer falls behind. This is reported
  per run.
- RNGD prefill runs in 128-token buckets (up to 1,024 tokens). Longer inputs use chunked prefill. The
  locally built 1B/3B artifacts accept at most 2,048 tokens of input plus output.
- tinyMMLU has 100 items. Its gp-IRT value estimates full-MMLU accuracy with an error of about 0.016 on MMLU in the
  tinyBenchmarks README; the raw 100-item accuracy moves by 0.01 per item. Near-ties between choices are common in
  BF16 (log-likelihoods move in steps of about 0.06–0.125), and each GPU–RNGD prediction difference observed so far
  involves a tie on at least one side.
- Scores are not comparable with the v1.0 MMLU 1,000-item generation-based scores.
