# LLM results: Llama 3 on a GPU system and a Furiosa RNGD system (unreleased)

> These are research benchmark results for two **whole systems** (runtime + driver + device), shown side by side.
> They are not chip results and not a ranking. All times are host-observed. Device-internal time is not
> observed. `max_observed` is the largest value in the measured population, not a worst-case bound. The MMLU
> numbers come from a zero-shot, generation-based evaluation. They are not comparable with published 5-shot
> log-likelihood MMLU scores.

This page summarizes the evidence behind the LLM path (Python measurement plane) for three models:

- Llama 3.2 1B Instruct
- Llama 3.2 3B Instruct
- Llama 3.1 8B Instruct

Each model ran on an NVIDIA GPU with vLLM and on a FuriosaAI RNGD card with furiosa-llm. The raw evidence bundles
are not in the repository, because they contain outputs derived from MMLU and host-local details. No model weight
and no MMLU question or answer is redistributed.

Every number below comes from the bundles and their bundle-only evaluations (catalog `sdv-bmt-llm-v1`), recorded
on 2026-11-17. Definitions, workloads and reproduction steps are in [`docs/llm_benchmark.md`](../llm_benchmark.md).

Statements on this page rest on one of three kinds of evidence, labeled where it matters:

- **Recorded:** written into a run's bundle while it ran. This covers timings, outputs, applied settings, code
  identity, configuration digest, SHA-256 of the input files and the tokenizer, RNGD artifact id, runtime
  versions and GPU UUID.
- **Separate check:** run once with a tool in `tools/llm/`, before or after the baselines. Its result file is kept
  with the local evidence. The versions of the tool and of the packages it ran with were not recorded.
- **Current-file comparison:** files as of 2026-11-17, after the runs, compared with the recorded values.

## Setup

| item | GPU system | RNGD system |
|---|---|---|
| device | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, one GPU (GPU 0; UUID recorded in every run) | FuriosaAI RNGD, one card (8 PEs); firmware 2026.4.0 recorded in every run |
| runtime (recorded) | vLLM 0.31.0 (V1 engine, `AsyncLLM`), torch 2.13.0+cu130; `max_model_len` 4,096, other engine settings at vLLM defaults (chunked prefill and CUDA graphs on) | furiosa-llm 2026.3.0, token-ID stream of the engine |
| precision | BF16, no quantization (recorded) | BF16 weights, activations and KV cache (artifact metadata) |
| model files | meta-llama Hugging Face revisions (table below) | 1B / 3B: built locally from the same revisions, batch 1, context 2,048. 8B: `furiosa-ai/Llama-3.1-8B-Instruct` at `b678d58b` (tag `v2026.3`) |
| controls (read back after loading) | greedy, prefix caching off, no speculative decoding, one request at a time, in-request KV cache on | same |
| host | shared multi-user host. Device idle was checked before every run. 1-min load average during the runs: 9.7–38.3 | same host. 1-min load average 35.8–57.8 |
| code identity (recorded) | Python source tree `a538e105…` for all 36 runs and their evaluations | same |

| model | meta-llama revision |
|---|---|
| Llama 3.2 1B Instruct | `9213176726f574b556790deb65791e0c5aa438b6` |
| Llama 3.2 3B Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Llama 3.1 8B Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |

**Inputs and weights.**

- **Recorded:** both systems received identical prompt token IDs for every MMLU item (1,000 / 1,000 per model).
  Every request stores the SHA-256 of its input token IDs.
- **Separate check (tokenizer files):** the RNGD 8B artifact's tokenizer gives every one of the 128,256 token IDs
  the same meaning as the meta-llama tokenizer. Its special tokens and chat template are also identical.
- **Separate check (weight files, before the runs):** every parameter tensor in the params files of the three
  RNGD artifacts is byte-identical to the local copy of the pinned meta-llama revision: 8B 291 / 291 (plus one
  rotary cos/sin table), 1B 145 / 145 and 3B 253 / 253 (tied embedding included).
- The bundles identify the weights by revision (GPU) and by artifact id (RNGD). The artifact id names the params
  file that was checked. The bundles hold no hash of the weight contents, so they link to the checked files by
  name and identifier only.

## 1. Speed (S1–S5, five runs per model and system)

Each workload is fixed-length synthetic input and output (input / output tokens): S1 128 / 64, S2 128 / 256,
S3 512 / 512, S4 830 / 256, S5 1,024 / 128. Each run sends 3 warm-up and 5 measured requests per workload, so
n = 25 per workload over five runs.

All 750 measured requests produced exactly the target number of tokens with finish reason `length`. There were no
length mismatches and no errors, and every run had prefix caching confirmed off.

Every measured request received exactly one token in its first stream item (n_first = 1). The RNGD system delivered
one token per stream item. On the GPU system, 9 of its 91,190 stream items carried more than one token: 8 for 1B and 1 for 3B.

**TTFT / TPOT, median (ms), n = 25 per cell**

| model | system | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | GPU · vLLM | 8.27 / 2.23 | 8.54 / 2.24 | 11.90 / 2.24 | 14.53 / 2.24 | 15.38 / 2.24 |
| Llama 3.2 1B | RNGD · furiosa-llm | 6.69 / 5.15 | 6.51 / 5.22 | 12.28 / 5.08 | 18.72 / 5.10 | 20.15 / 5.16 |
| Llama 3.2 3B | GPU · vLLM | 12.14 / 5.35 | 12.69 / 5.36 | 23.24 / 5.39 | 32.55 / 5.41 | 35.33 / 5.41 |
| Llama 3.2 3B | RNGD · furiosa-llm | 12.33 / 9.80 | 12.61 / 9.84 | 24.73 / 9.83 | 43.05 / 9.21 | 43.13 / 9.24 |
| Llama 3.1 8B | GPU · vLLM | 20.96 / 10.98 | 21.36 / 11.04 | 43.34 / 11.13 | 64.89 / 11.16 | 71.70 / 11.17 |
| Llama 3.1 8B | RNGD · furiosa-llm | 21.00 / 15.69 | 20.89 / 15.80 | 45.56 / 15.86 | 84.47 / 15.84 | 87.11 / 15.83 |

**S3 (512 / 512) in detail (ms), n = 25**

| model | system | TTFT p50 | TTFT p90 | TTFT min–max | TPOT p50 | TPOT p90 | TPOT min–max | decode tokens/s p50 | run-to-run spread of TPOT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | GPU · vLLM | 11.90 | 13.08 | 10.56–15.03 | 2.24 | 2.25 | 2.24–2.26 | 446 | 0.3% |
| Llama 3.2 1B | RNGD · furiosa-llm | 12.28 | 13.57 | 11.13–13.80 | 5.08 | 5.43 | 4.54–5.64 | 197 | 9.9% |
| Llama 3.2 3B | GPU · vLLM | 23.24 | 25.77 | 20.11–26.95 | 5.39 | 5.40 | 5.36–5.40 | 186 | 0.6% |
| Llama 3.2 3B | RNGD · furiosa-llm | 24.73 | 26.08 | 23.65–26.90 | 9.83 | 10.01 | 8.70–10.07 | 102 | 10.9% |
| Llama 3.1 8B | GPU · vLLM | 43.34 | 45.40 | 40.26–46.21 | 11.13 | 11.15 | 11.08–11.15 | 90 | 0.5% |
| Llama 3.1 8B | RNGD · furiosa-llm | 45.56 | 46.04 | 44.21–46.95 | 15.86 | 16.08 | 15.52–16.20 | 63 | 2.8% |

Percentiles are nearest-rank. p99 is not reported, because it needs at least 100 samples. The run-to-run spread is
(largest − smallest of the five per-run TPOT medians) ÷ (median of those five medians).

What the tables show, without attributing a cause:

- With short inputs (S1, S2), the two systems reach the first token in similar times.
- With 830–1,024 input tokens (S4, S5), the RNGD system takes longer to the first token.
- Time per output token is shorter on the GPU system for all three models.
- Across S1–S5, the five per-run TPOT medians differ by less than 1% on the GPU system and by 2.5–14.2% on the
  RNGD system (same spread definition as above).

The cause of the RNGD run-to-run spread was not determined. The RNGD runs took place while the host load was
higher, but nothing in the evidence ties the spread to that load.

## 2. Quality: MMLU 1,000 items, zero-shot generation-based evaluation

The item set is `cais/mmlu` (`all`, test split, revision `c30699e8`): 1,000 items, chosen in proportion to subject
size. Each item is one user message in the model's chat template. Decoding is greedy, with at most 16 new tokens.

Every item ran on both systems (coverage 1.0). An unparsable output counts as wrong and is also reported in its own
column.

| model | system | correct | wrong (parsed) | unparsable | accuracy |
|---|---|---:|---:|---:|---:|
| Llama 3.2 1B | GPU · vLLM | 342 | 512 | 146 | 0.342 |
| Llama 3.2 1B | RNGD · furiosa-llm | 341 | 516 | 143 | 0.341 |
| Llama 3.2 3B | GPU · vLLM | 572 | 393 | 35 | 0.572 |
| Llama 3.2 3B | RNGD · furiosa-llm | 574 | 391 | 35 | 0.574 |
| Llama 3.1 8B | GPU · vLLM | 644 | 343 | 13 | 0.644 |
| Llama 3.1 8B | RNGD · furiosa-llm | 641 | 348 | 11 | 0.641 |

The criterion is |accuracy(RNGD) − accuracy(GPU)| ≤ 0.01 on this fixed item set. It measures total quality only.
It does not mean the two systems compute the same numbers or produce the same output per item.

| model | difference (RNGD − GPU) | verdict | GPU correct → RNGD wrong | GPU wrong → RNGD correct | both parsed, different letters | prompt token IDs |
|---|---:|---|---:|---:|---:|---|
| Llama 3.2 1B | −0.001 | pass | 9 | 8 | 29 | identical |
| Llama 3.2 3B | +0.002 | pass | 5 | 7 | 16 | identical |
| Llama 3.1 8B | −0.003 | pass | 7 | 4 | 16 | identical |

Most unparsable outputs begin with an explanation instead of a letter and reach the 16-token limit.

## 3. Checks

| check | kind | result |
|---|---|---|
| prefix caching off in the measured runs | recorded | applied setting off, per-request reused tokens 0 and hit counters unchanged in all 36 runs |
| prefix-cache positive control | separate check (vLLM with 1B, furiosa-llm with 8B) | vLLM with caching on: the second identical request reused 1,008 tokens and the hit counter rose by 1,008. With caching off: 0 / 0. furiosa-llm with caching on: 1,023 reused tokens and 1,026 hits. With caching off: 0. Timing was used only as a diagnostic |
| GPU path vs Hugging Face transformers (FP32, CPU), same prompt token IDs | separate check; the MMLU part compares with the recorded GPU MMLU outputs | criteria: first token ≥ 15 / 16 (any mismatch within the HF top 2), identical 32-token greedy output ≥ 12 / 16, MMLU answer agreement on the first 100 items ≥ 95. 1B: 16 / 16, 13 / 16, 95. 3B: 15 / 16, 13 / 16, 98. 8B: 16 / 16, 15 / 16, 99. All pass. The 1B MMLU agreement is exactly at the criterion |
| independent re-scoring | separate check | a separately written scorer (`tools/llm/rescore_mmlu.py`) matched the evaluator item by item in all 6 MMLU runs |
| re-evaluation with the published code | current code on the recorded bundles | all 36 bundles evaluated again: metrics content identical in 36 / 36. The 9 summaries recomputed from them were also identical |
| current files vs recorded values | current-file comparison | experiment configurations (36 / 36), input files, `tokenizer.json`, RNGD artifact ids (3 / 3) and runtime versions equal the recorded values |

## 4. Limits

- Times are host-observed times of the whole system. Device-internal time is not observed.
- The RNGD path reads token IDs from the furiosa-llm engine stream that sits under the public text-streaming API.
  That interface is not in the public furiosa-llm documentation. Its behavior was checked in smoke runs and in
  every baseline run.
- The RNGD 1B and 3B artifacts were built for batch 1 and a 2,048-token context. The 8B artifact is the published
  build for larger batches and contexts. Keep this difference in mind when comparing models on the RNGD system.
- With n = 25 per workload there is no p99.
- The host was shared and loaded during the runs. The load was logged before every run.
- The bundles do not record the following:
  - the versions of the tools in `tools/llm/` used for the separate checks;
  - the `sdv_bmt` code used by the GPU vs Hugging Face check;
  - a hash of the weight file contents;
  - the full package lists of the environments.

  No identity beyond the recorded values is claimed for these. The environment lists and what was actually
  installed are described in `docs/llm_benchmark.md`, section 4.
