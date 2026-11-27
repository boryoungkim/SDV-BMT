# LLM results: Llama 3 on a GPU system and a Furiosa RNGD system (unreleased)

> These are research benchmark results for two **whole systems** (runtime + driver + device), shown side by side.
> They are not chip results and not a ranking. All times are host-observed. Device-internal time is not
> observed. `max_observed` is the largest value in the measured population, not a worst-case bound. Quality is
> tinyMMLU (100 items) evaluated as the lm-evaluation-harness task `tinyMMLU`. Its official metric is the IRT++
> accuracy estimate, the value the harness reports; the raw accuracy over the 100 items is shown next to it.

This page summarizes the evidence behind the LLM path (Python measurement plane) for three models:

- Llama 3.2 1B Instruct
- Llama 3.2 3B Instruct
- Llama 3.1 8B Instruct

Each model ran on an NVIDIA GPU with vLLM and on a FuriosaAI RNGD card with furiosa-llm. The raw evidence bundles
are not in the repository, because they contain outputs derived from tinyMMLU (answer letters and log-likelihoods)
and host-local details. No model weight and no tinyMMLU question or answer is redistributed.

Every number below comes from the bundles and their bundle-only evaluations (catalog `sdv-bmt-llm-v1`). All runs
were recorded on 2026-11-27 from an unmodified checkout of commit `cf716c4`: 15 speed runs and 3 tinyMMLU runs per
system. The tinyMMLU runs repeat those of 2026-11-25 (commit `7a43714`) with identical results (section 2).
Definitions, workloads and reproduction steps are in [`docs/llm_benchmark.md`](../llm_benchmark.md).

Statements on this page rest on one of two kinds of evidence, labeled where it matters:

- **Recorded:** written into a run's bundle while it ran. This covers timings, outputs, applied settings, code
  identity, configuration digest, SHA-256 of the input files and the tokenizer, RNGD artifact id, runtime
  versions and GPU UUID.
- **Separate check:** run once with a tool in `tools/llm/`, before or after the baselines. Its result file is kept
  with the local evidence. The versions of the tool and of the packages it ran with were not recorded.

## Setup

| item | GPU system | RNGD system |
|---|---|---|
| device | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, one GPU (GPU 0; UUID recorded in every run) | FuriosaAI RNGD, one card (8 PEs); firmware 2026.4.0 recorded in every run |
| runtime (recorded) | vLLM 0.31.0 (V1 engine, `AsyncLLM`), torch 2.13.0+cu130; `max_model_len` 4,096, other engine settings at vLLM defaults (chunked prefill and CUDA graphs on) | furiosa-llm 2026.3.0, token-ID stream of the engine |
| precision | BF16, no quantization (recorded) | BF16 weights, activations and KV cache (artifact metadata) |
| model files | meta-llama Hugging Face revisions (table below) | 1B / 3B: built locally from the same revisions, batch 1, context 2,048 (speed) and 4,096 (tinyMMLU, whose longest request is 3,029 tokens). 8B: `furiosa-ai/Llama-3.1-8B-Instruct` at `b678d58b` (tag `v2026.3`), both workloads |
| controls (read back after loading) | greedy, prefix caching off, no speculative decoding, one request at a time, in-request KV cache on | same |
| host | shared multi-user host. Each model started only after the host was quiet (fewer than 4 busy CPUs over 30 s, three times in a row) and the host's other GPU was idle; device idle was checked before every run. 1-min load average at run start: speed runs 1.2–5.8, tinyMMLU runs 2.9–4.7 | same host and rule. Speed runs 2.5–17.3, tinyMMLU runs 5.4–6.7 |
| code identity (recorded) | Python source tree `e6376ee7…` (commit `cf716c4`) for the 15 speed runs and the 3 tinyMMLU runs, and their evaluations | same (15 speed and 3 tinyMMLU runs) |

| model | meta-llama revision |
|---|---|
| Llama 3.2 1B Instruct | `9213176726f574b556790deb65791e0c5aa438b6` |
| Llama 3.2 3B Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Llama 3.1 8B Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |

**Inputs and weights.**

- **Recorded:** both systems received identical prompt token IDs for every tinyMMLU request (400 / 400 per model).
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
one token per stream item. On the GPU system, 6 of its 91,194 stream items carried more than one token, all for 1B.

**TTFT / TPOT, median (ms), n = 25 per cell**

| model | system | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | GPU · vLLM | 8.91 / 2.23 | 10.86 / 2.25 | 14.07 / 2.25 | 16.52 / 2.25 | 16.90 / 2.25 |
| Llama 3.2 1B | RNGD · furiosa-llm | 7.70 / 5.66 | 7.50 / 5.60 | 13.23 / 5.77 | 19.86 / 5.71 | 21.12 / 5.85 |
| Llama 3.2 3B | GPU · vLLM | 13.64 / 5.35 | 14.42 / 5.35 | 23.35 / 5.38 | 32.24 / 5.38 | 35.04 / 5.39 |
| Llama 3.2 3B | RNGD · furiosa-llm | 13.34 / 10.48 | 13.30 / 10.33 | 26.10 / 10.39 | 44.49 / 10.35 | 45.13 / 10.45 |
| Llama 3.1 8B | GPU · vLLM | 22.53 / 10.92 | 23.81 / 10.98 | 45.14 / 11.05 | 65.81 / 11.07 | 72.62 / 11.08 |
| Llama 3.1 8B | RNGD · furiosa-llm | 21.85 / 16.57 | 21.92 / 16.45 | 46.22 / 16.43 | 85.28 / 16.41 | 87.41 / 16.33 |

**S3 (512 / 512) in detail (ms), n = 25**

| model | system | TTFT p50 | TTFT p90 | TTFT min–max | TPOT p50 | TPOT p90 | TPOT min–max | decode tokens/s p50 | run-to-run spread of TPOT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | GPU · vLLM | 14.07 | 14.81 | 11.88–15.75 | 2.25 | 2.27 | 2.23–2.27 | 444 | 1.6% |
| Llama 3.2 1B | RNGD · furiosa-llm | 13.23 | 13.78 | 11.31–14.07 | 5.77 | 6.02 | 5.52–6.08 | 173 | 2.4% |
| Llama 3.2 3B | GPU · vLLM | 23.35 | 24.85 | 22.11–25.86 | 5.38 | 5.38 | 5.35–5.39 | 186 | 0.6% |
| Llama 3.2 3B | RNGD · furiosa-llm | 26.10 | 26.66 | 25.34–26.70 | 10.39 | 10.47 | 10.06–10.64 | 96 | 1.5% |
| Llama 3.1 8B | GPU · vLLM | 45.14 | 46.35 | 41.63–47.27 | 11.05 | 11.07 | 11.01–11.07 | 90 | 0.4% |
| Llama 3.1 8B | RNGD · furiosa-llm | 46.22 | 46.67 | 44.42–46.90 | 16.43 | 16.50 | 16.18–16.75 | 61 | 0.4% |

Percentiles are nearest-rank. p99 is not reported, because it needs at least 100 samples. The run-to-run spread is
(largest − smallest of the five per-run TPOT medians) ÷ (median of those five medians).

What the tables show, without attributing a cause:

- With short inputs (S1, S2), the RNGD system reaches the first token as fast as or faster than the GPU system.
- With 830–1,024 input tokens (S4, S5), the RNGD system takes longer to the first token.
- Time per output token is shorter on the GPU system for all three models.
- Across S1–S5, the five per-run TPOT medians differ by 0.4–1.6% on the GPU system and by 0.4–12.8% on the RNGD
  system (same spread definition as above).

The cause of the RNGD run-to-run spread was not determined.

**Earlier speed runs.** An earlier set of the same speed runs (2026-11-17, an earlier commit, host 1-min load
9.7–57.8) gave GPU TPOT within 1% of the values above. The other medians were lower then: GPU TTFT by 1–27%
(except 3B S3–S5, within 1%), RNGD TTFT by 0–15% and RNGD TPOT by 3–14%. The cause of these differences was not
determined, so neither set is preferred as more typical of the systems.

## 2. Quality: tinyMMLU, 100 items (lm-evaluation-harness method)

The item set is `tinyBenchmarks/tinyMMLU` (`all`, test split, revision `cd6a8ee9`): all 100 items. Each item's
`input_formatted` text is the prompt, unchanged: the Open LLM Leaderboard MMLU prompt with five solved examples
already inside, no chat template. Each answer choice `" A"` … `" D"` is one request. Its log-likelihood comes from
the runtime's prompt log-probabilities, with the harness's tokenization, and the prediction is the first choice with
the largest log-likelihood (`acc_norm` equals `acc` here). Details: [`docs/llm_benchmark.md`](../llm_benchmark.md),
section 2.

One run per model and system. The method has no sampling, so a repeat gives the same numbers: the six runs of
2026-11-27 (1-min host load 2.9–6.7), those of 2026-11-25 (43–76) and those of 2026-11-21 (24–65), made with
different code checkouts of the same method, have identical prompt log-probabilities for every request (403 / 403
per run, warm-up included). The host
load therefore does not enter these numbers. Every run scored all 100 items (all 400 requests returned
log-probabilities).

| model | system | tinyMMLU accuracy, IRT++ estimate | raw accuracy (correct of 100) |
|---|---|---:|---:|
| Llama 3.2 1B | GPU · vLLM | 0.4472 | 0.43 (43) |
| Llama 3.2 1B | RNGD · furiosa-llm | 0.4354 | 0.42 (42) |
| Llama 3.2 3B | GPU · vLLM | 0.6305 | 0.60 (60) |
| Llama 3.2 3B | RNGD · furiosa-llm | 0.6316 | 0.59 (59) |
| Llama 3.1 8B | GPU · vLLM | 0.6299 | 0.61 (61) |
| Llama 3.1 8B | RNGD · furiosa-llm | 0.6346 | 0.63 (63) |

The IRT++ estimate is the official tinyMMLU metric: the tinyMMLU dataset card calls it the best accuracy
estimator, and the lm-evaluation-harness task returns it. It is computed from the per-item correctness vector by
`tools/llm/tinymmlu_irt.py` with the pinned tinyBenchmarks IRT parameters (package key `gpirt`); the tinyBenchmarks
README gives its error on MMLU as about 0.016. The raw accuracy moves by 0.01 per item; the GPU–RNGD comparison
below works on the items behind it.

**GPU–RNGD comparison.** In BF16 the choice scores move in steps of 0.0625–0.125 for these models, so when the best
two choices are one or two steps apart, rounding alone can flip the answer. The comparison
(`python -m sdv_bmt.llm compare`) therefore uses a near-tie rule: an item is *near-tie* when, on either system, the
best and second-best scores are at most 0.25 apart; every other item is *clear*. The result passes when the two
systems give the same answer on every clear item. Near-tie items are counted, not judged.

| model | accuracy difference (RNGD − GPU) | clear items | clear answers that differ | near-tie items | near-tie answers that differ | largest score difference | prompt token IDs | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Llama 3.2 1B | −0.01 | 71 | 0 | 29 | 2 | 0.414 | identical | pass |
| Llama 3.2 3B | −0.01 | 79 | 0 | 21 | 3 | 0.248 | identical | pass |
| Llama 3.1 8B | +0.02 | 81 | 0 | 19 | 4 | 0.358 | identical | pass |

The verdict says the two systems pick the same answer wherever the answer is not decided by rounding. It is not a
statement that they compute the same numbers. The accuracy differences come from near-tie items only.

## 3. Checks

| check | kind | result |
|---|---|---|
| prefix caching off in the measured runs | recorded | applied setting off and per-request reused tokens 0 in all 30 speed runs and all 6 tinyMMLU runs; hit counters unchanged in the speed runs |
| prefix-cache positive control | separate check (vLLM with 1B, furiosa-llm with 8B) | vLLM with caching on: the second identical request reused 1,008 tokens and the hit counter rose by 1,008. With caching off: 0 / 0. furiosa-llm with caching on: 1,023 reused tokens and 1,026 hits. With caching off: 0. Timing was used only as a diagnostic |
| GPU path vs Hugging Face transformers (FP32, CPU), same prompt token IDs | separate check (2026-11-17) | criteria: first token ≥ 15 / 16 (any mismatch within the HF top 2), identical 32-token greedy output ≥ 12 / 16. 1B: 16 / 16, 13 / 16. 3B: 15 / 16, 13 / 16. 8B: 16 / 16, 15 / 16. All pass |
| tinyMMLU path vs lm-evaluation-harness 0.4.13 itself (vLLM, GPU) | separate check (`tools/llm/tinymmlu_harness_check.py`) | the harness reads the same 100 documents and builds the same token IDs for all 400 requests. Predictions differ on 0–3 items per model and setting, each a tie on one side and a one-step BF16 margin (0.125) on the other; Llama 3.2 1B under batch 1 and prefix caching off: identical predictions and identical reported value |
| code identity | recorded | the same Python source tree (`e6376ee7…`, commit `cf716c4`) in all 36 runs and in all their evaluations |

## 4. Limits

- Times are host-observed times of the whole system. Device-internal time is not observed.
- The RNGD path reads token IDs from the furiosa-llm engine stream that sits under the public text-streaming API.
  That interface is not in the public furiosa-llm documentation. Its behavior was checked in smoke runs and in
  every baseline run.
- The RNGD 1B and 3B artifacts were built for batch 1 and a 2,048-token context (speed) or a 4,096-token context
  (tinyMMLU). The 8B artifact is the published build for larger batches and contexts. Keep this difference in mind
  when comparing models on the RNGD system.
- With n = 25 per workload there is no p99.
- The host was shared. Each model started only when the host was quiet, and the load was logged before every
  run; the load during a run was not observed.
- The bundles do not record the following:
  - the versions of the tools in `tools/llm/` used for the separate checks;
  - the `sdv_bmt` code used by the GPU vs Hugging Face check;
  - a hash of the weight file contents;
  - the full package lists of the environments.

  No identity beyond the recorded values is claimed for these. The environment lists and what was actually
  installed are described in `docs/llm_benchmark.md`, section 4.
