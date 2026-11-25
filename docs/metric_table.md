# Metric table (`metrics.csv`)

The metric table is one CSV file per call that holds the benchmark result of one or more models, metric by metric.
It is made from evaluated runs after `run` and `evaluate`. It measures nothing and evaluates nothing.

## Running a model with one command

`tools/run_model.sh` runs one model end to end on its host: validation, the timed runs, evaluation and the metric
table. The GPU / RNGD models run on the reference host, the Mobilint MLA400 models on the host of the MLA400 card.

```sh
tools/run_model.sh list                 # the model names (12 on the GPU, 3 on the RNGD card, 5 on the MLA400)
tools/run_model.sh yolox-s              # one model (validate, 5 runs, evaluate, table)
tools/run_model.sh yolox-s --dry-run    # print the commands only; uses no device and writes nothing
tools/run_model.sh all                  # the 12 models on the GPU in order; a failed model does not stop the others
tools/run_model.sh centerpoint-pillar02-mini   # CenterPoint on the 162 mini samples only (no val run series)
tools/run_model.sh all-mla400           # the 5 MLA400 models in order, on the MLA400 host
```

| Variable | Meaning |
|---|---|
| `R` | results root (default `runs/bench-YYYYMMDD`); a model whose runs already exist below it is refused |
| `REPS` | runs per model (default 5); the tinyMMLU runs and the CenterPoint val series run once |
| `SDV_BMT_MODELS`, `SDV_BMT_DATASETS`, `SDV_BMT_CACHE`, `VLLM_PY`, `FURIOSA_PY`, `HF_HOME` | Llama models only, see `docs/llm_benchmark.md` |
| `SDV_BMT_PREPARED_INPUT_CACHE` | optional: a directory for the prepared-input cache (`runner/prepcache.py`). Later runs reuse verified `Workload.preprocess` results of the same code, input contract and items (YOLOX on 5,000 images: about 12 minutes of preprocessing per run otherwise). Prefer a fast local disk; about 25 GB for the 640 input and 10 GB for the 416 input |
| `LMEVAL_PY`, `TINYMMLU_IRT_PKL` | Llama models, optional: the lm-eval environment and the pinned `tinyBenchmarks.pkl`; with both, the table also carries the tinyMMLU gp-IRT value (`tools/llm/tinymmlu_irt.py`) |
| `MLA400_DEVICE` | MLA400 models only: the device file checked for other users (default: the device of the DUT config) |

What each model runs:

| Models | Runs | Evaluation catalog |
|---|---|---|
| `yolox-{tiny,s,m,l,x}`, `deeplabv3plus-{r18,r50,r101}` | `configs/experiments/<model>-baseline.yaml`, `REPS` runs | `sdv-bmt-model-metric-set-v1.1` |
| `centerpoint-pillar02` | `centerpoint-pillar02-mini-baseline.yaml` (`REPS` runs) and the run series `centerpoint-pillar02-val-series.yaml` (once) | `sdv-bmt-model-metric-set-v1.2` |
| `centerpoint-pillar02-mini` | `centerpoint-pillar02-mini-baseline.yaml` only (`REPS` runs on the 162 mini samples, no val run series) | `sdv-bmt-model-metric-set-v1.2` |
| `llama-3.2-1b-instruct`, `llama-3.2-3b-instruct`, `llama-3.1-8b-instruct` | on the GPU: `tools/llm/run_baselines.sh gpu`, `REPS` speed runs and one tinyMMLU run | `sdv-bmt-llm-v1` |
| `llama-3.2-1b-instruct-rngd`, `llama-3.2-3b-instruct-rngd`, `llama-3.1-8b-instruct-rngd` | on the FuriosaAI RNGD card: `tools/llm/run_baselines.sh rngd`, `REPS` speed runs and one tinyMMLU run | `sdv-bmt-llm-v1` |
| `yolox-{tiny,s,m,l,x}-mla400` | `configs/experiments/<model>-baseline.yaml` on the Mobilint MLA400, `REPS` runs | `sdv-bmt-model-metric-set-v1.1` |

A Llama model runs on one device per name, and its table holds that device only (`tables/llama-<m>.metrics.csv` for
the GPU, `tables/llama-<m>-rngd.metrics.csv` for the RNGD card). Existing runs are refused per device, so the GPU and
RNGD runs of a model can share one results root.

Before an MLA400 model starts, the script checks that the native runner `sdv-bmt-native-mobilint` is built
(`tools/mobilint/build_native.sh`), that the device node of the DUT config exists and that the compiled models
(`.mxq`) named by the workloads are present (`tools/mobilint/README.md`). Before every MLA400 run it stops while
another process holds the device open. The Mobilint SDK license forbids publishing MLA400 performance, accuracy or
benchmark results without Mobilint's written consent (`docs/mobilint/mla400-yolox-port.md`): keep their tables in
the results root.

The script starts with an SDV-BMT banner, the results root, the models and their datasets (read from the plans,
written as the dataset line of the result table), and prints a header line per model. While it runs,
it shows progress: the run number and a bar of the elapsed time against the typical duration on the reference
host (an estimate, held at 99 % until the run ends), with the finished parts of the run series and the finished
Llama runs. When a model is done it prints the headline table of
the model and the path of its CSV file; `all` ends with a summary of the 12 models (status, time, rows, CSV path). It holds
`runs/TIMED_RUN_IN_PROGRESS` during CNN / LiDAR runs (`run_baselines.sh` does so for Llama) and refuses to start
while another timed run holds it. Output below `$R`:

```text
vision/<experiment>/run-*/ ...      CNN / LiDAR run bundles and their evaluations
llm/<experiment>/run-*/ ...         LLM run bundles and their evaluations
logs/<model>.log                    everything the commands printed
tables/<model>.metrics.csv          the metric table
tables/<model>.console.txt          the representative metrics as printed
```

To read a table, `tools/show_metrics.py --headline FILE.csv [...]` prints the row counts and headline rows of each
file, and `tools/show_metrics.py FILE.csv` every model row per experiment and layer, with times in milliseconds and
the status counts. It only reads the files.

## Making a table from evaluated runs

| Models | Command |
|---|---|
| CNN and LiDAR workloads (YOLOX, DeepLabV3+, CenterPoint) | `python -m sdv_bmt.cli.table <experiment dir or run series dir> [...] --out <metrics.csv>` |
| LLM workloads (Llama 3, GPU and RNGD) | `python -m sdv_bmt.llm table <LLM experiment dir> [...] --out <metrics.csv>` |

An experiment directory holds `experiment.json` and its run directories. A run series directory is the `series-*`
directory that holds `series.json`. Pass one directory for the table of one model, or several in the order the
table should list them. The output file must not exist.

Both commands print the representative metrics to the console (stderr): one framed table per model (per model and
device for an LLM) that names the task, dataset, device and experiment, with the columns Metric (readable name), Value, Unit, Valid runs and Min – Max, followed by the
absolute path of the CSV file that holds every metric and every run. Stdout carries a short JSON summary (file,
SHA-256, row counts).

```text
┌─ yolox-s-coco-v1-quality ────────────────────────────────────────────────────
│ task         coco-2017-object-detection
│ dataset      coco-val2017 (5,000 items)
│ device       NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
│ experiment   yolox-s-baseline
│
│ Metric                              Value  Unit   Valid runs  Min – Max
│ ─────────────────────────────────  ──────  ─────  ──────────  ───────────────
│ DUT service time (median)           4.364  ms            5/5  4.360 – 4.380
│ Output-ready latency (p99)          4.654  ms            5/5  4.643 – 4.865
│ Deadline miss rate (output ready)  0.0000  ratio         5/5  0.0000 – 0.0000
│ COCO box AP                        0.4289  ratio         5/5  0.4289 – 0.4289
└──────────────────────────────────────────────────────────────────────────────

Value = median of the valid runs; Min – Max = range of the valid run values.
For detailed results (every metric, every run), see the CSV file:
  /path/to/runs/bench-YYYYMMDD/tables/yolox-s.metrics.csv
```

## Layout

1. **Headline rows first** (`row = headline`). Four representative metrics of every model under a readable name
   (`metric_name`); times in milliseconds.
2. **Model rows** (`row = model`). One row per experiment and metric: the overall result of the model over its runs.
3. **Run rows** (`row = run`). One row per run and metric, copied unchanged from the newest evaluation of the run.
   A run series has one evaluation and gives one set of run rows, identified by its series identifier.

| Column | Meaning |
|---|---|
| `row` | `headline`, `model` or `run` |
| `task` | task of the workload (`coco-2017-object-detection`, `cityscapes-semantic-segmentation-19class`, `nuscenes-3d-detection-10class`, `llm-speed`, `llm-tinymmlu`; `llm-mmlu` for v1.0 bundles) |
| `dataset` | input data of the run. CNN / LiDAR: the input set of `manifest.json` and its item count, for example `coco-val2017 (5,000 items)` (a run series: the items of all parts). LLM: `speed-s1-s5 (synthetic token IDs)` for the speed workload file, `tinymmlu-100 (100 items)` for the tinyMMLU item list with the item total of the evaluation |
| `model` | model (workload) name |
| `device` | device name as recorded in the run: the GPU product name, or vendor and accelerator family with the DUT config (for example `Mobilint NPU (mobilint-mla400-aries2-single-c0c0)`) |
| `experiment_id` | experiment identifier |
| `run_id` | run identifier (a run series: its series identifier); empty on headline and model rows |
| `eval_id` | evaluation the run row was copied from; empty on headline and model rows |
| `catalog_version` | metric catalog of the evaluation |
| `metric_name` | headline rows: readable name; empty otherwise |
| `metric_id`, `layer`, `unit` | metric identifier (on a headline row: the metric it repeats), result layer, unit (empty for a text value) |
| `status` | run row: status in the evaluation; headline and model rows: see below |
| `value` | run row: the evaluated value; headline and model rows: see below |
| `min`, `max` | headline and model rows, numeric metric: smallest and largest valid run value |
| `n_runs`, `n_valid` | headline and model rows: runs that carry the metric, runs whose value is valid |

Use a value only when its `status` is `valid`. Times are integers in nanoseconds except on headline rows (ms);
empty cells mean null or not applicable.

## Headline metrics

| Task | Headline rows (`metric_name` ← `metric_id`) |
|---|---|
| 2D detection (YOLOX) | DUT service time (median) ← `bmt.dut_service.p50`; Output-ready latency (p99) ← `bmt.output_ready.p99`; Deadline miss rate (output ready) ← `bmt.deadline.output_ready_miss_rate`; COCO box AP ← `quality.detection.conditional.ap` |
| Segmentation (DeepLabV3+) | the same three time metrics; Cityscapes mIoU (19 classes) ← `quality.segmentation.conditional.miou` |
| 3D detection (CenterPoint) | the same three time metrics; nuScenes detection score (NDS) ← `quality.det3d.conditional.nds` |
| LLM, per device | Time to first token, TTFT (S3, median) ← `llm.S3.ttft_ns.p50`; Time per output token, TPOT (S3, median) ← `llm.S3.tpot_ns.p50`; End-to-end latency (S3, median) ← `llm.S3.e2e_ns.p50` (speed experiment); tinyMMLU accuracy (100 items, lm-evaluation-harness method) ← `llm.tinymmlu.accuracy` and tinyMMLU gp-IRT estimate of MMLU accuracy ← `llm.tinymmlu.gpirt` (tinyMMLU experiment; the raw 100-item accuracy and the value the harness reports, side by side). S3 is the speed workload with 512 input and 512 output tokens |

A headline row repeats the model row of its metric; the only change is ns → ms (value / 10^6). A headline metric the
evaluation does not contain is listed with `status = not_in_evaluation`.

## Model rows

Model rows are the only values the table forms. They describe the runs and are not a statistical test.

| Run values of the metric | Model row |
|---|---|
| numeric, at least one valid | `value` = median of the valid run values, with `min`, `max`; `status = valid` |
| text, every run valid and equal | `value` = that text; `status = valid` |
| text, different between runs | `status = differs_across_runs` |
| none valid | `status` = the status all runs share (for example `insufficient_samples`), otherwise `no_valid_run` |

A run value counts only when its status is `valid`. With one run (a tinyMMLU run, a run series) the model row repeats
the run value.

## Rules

- **Copied, not computed.** Run rows are the evaluation records unchanged. The newest evaluation of every run is
  used (latest `created_at_utc` in `eval_manifest.json`, ties: the larger eval_id).
- **Checked before writing.** The commands refuse a raw file that differs from its `manifest.json`, a run series
  part whose manifest differs from `series.json`, a run without an evaluation, an evaluation whose `metrics.json`
  differs from its `eval_manifest.json` record, an evaluation made from a different `manifest.json` or
  `series.json`, runs of one experiment made from different plans (LLM: different configs), the same experiment
  twice, and MOCK and ACTUAL runs in one table. Nothing is written when a check fails.
- **Deterministic.** The same inputs give byte-identical files.
- **Catalogs differ.** YOLOX evaluations of v0.1.2 use `sdv-bmt-model-metric-set-v1` (`tools/run_model.sh` uses
  `v1.1`, which adds two segmentation metrics that are `not_applicable` for detection), DeepLabV3+ `v1.1`,
  CenterPoint `v1.2` and the LLM path `sdv-bmt-llm-v1`. A metric of the same name can differ in definition between
  catalogs; compare values only within one catalog.

## LLM metric names

An LLM evaluation (`metrics.json`, catalog `sdv-bmt-llm-v1`) is nested. The table spreads it into metric rows; the
values are copied, nothing is recomputed.

| Metric | Source in `metrics.json` | Layer, unit |
|---|---|---|
| `llm.run.status` | `run_status.run_status` | validity_evidence, text |
| `llm.count.{planned,completed,error}` | `run_status.request_counts` | performance, count |
| `llm.control.prefix_caching` | `controls.prefix_caching.verdict` | validity_evidence, text |
| `llm.control.one_request_at_a_time` | `controls.concurrency.status` | validity_evidence, text |
| `llm.streaming.unit` | `streaming.unit` (`token` or `chunk`) | validity_evidence, text |
| `llm.<S1..S5>.count.{measured,fixed_length_ok,length_mismatch,error}` | `speed.<S>.counts` | performance, count |
| `llm.<S1..S5>.<ttft_ns, tpot_ns, e2e_ns, decode_tps>.<p50, p90, p99, min, max_observed>` | `speed.<S>.<quantity>` (`p50` = `median_p50`) | performance, ns or 1/s |
| `llm.tinymmlu.{accuracy,coverage}` | `quality.accuracy`, `quality.coverage` | task_quality, ratio |
| `llm.tinymmlu.count.{total,correct,incorrect,failed,not_run}` | `quality.total`, `quality.counts` | task_quality, count |
| `llm.tinymmlu.gpirt` | `estimate.gpirt` of the run in a result file of `tools/llm/tinymmlu_irt.py` given with `--irt` (run and evaluation IDs must match the table's evaluation; IRT parameter file pinned); `not_computed` without one | task_quality, ratio |
| `llm.tinymmlu.<ttft_ns, e2e_ns>.<p50, p90, p99, min, max_observed>` | `task_request_times.<quantity>` (one request per item and choice) | performance, ns |
| `llm.mmlu.{accuracy,coverage}` (v1.0 bundles) | `quality.accuracy`, `quality.coverage` | task_quality, ratio |
| `llm.mmlu.count.{total,correct,incorrect_parsable,unparsable,failed,not_run}` | `quality.total`, `quality.counts` | task_quality, count |
| `llm.mmlu.<ttft_ns, e2e_ns>.<p50, p90, p99, min, max_observed>` | `mmlu_request_times.<quantity>` | performance, ns |

A percentile keeps its own status: with 5 requests per speed workload and run, `p99` is `insufficient_samples`
(p99 needs at least 100 samples).

The model row of an LLM speed metric is the median of the per-run values (for example the median of the five
per-run TTFT medians). The LLM result page uses `python -m sdv_bmt.llm aggregate`, which pools the requests of the
five runs before taking the median (n = 25), so the two numbers can differ slightly.
