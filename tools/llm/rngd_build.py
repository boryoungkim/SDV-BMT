"""Build a furiosa-llm (2026.3, legacy artifact path) RNGD artifact for a local Llama checkpoint.

Bucket set = the SDK's own Llama-3.1-8B preset restricted to batch 1 and context <= MAX_MODEL_LEN,
so 1B/3B use the same prefill / append / tokenwise structure as the pre-compiled 8B artifact.
The `furiosa-llm build` CLI cannot pass append buckets (needed for prompts > 1024 tokens), so this
script calls the same ArtifactBuilder the CLI uses. Run with the furiosa-llm environment's
interpreter (the environment is not modified):

    <venv>/bin/python tools/llm/rngd_build.py --model <hf dir> --out <artifact dir> --cache-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MAX_MODEL_LEN = 2048


def bucket_config_dict(max_len: int = MAX_MODEL_LEN) -> dict:
    from furiosa_llm.artifact.presets import LLAMA_3_1_8B_PRESET as P

    return {
        "prefill_buckets": [b for b in P.prefill_buckets if b[0] == 1 and b[1] <= max_len],
        "decode_buckets": [b for b in P.decode_buckets if b[0] == 1 and b[1] <= max_len],
        "append_buckets": [b for b in P.append_buckets if b[0] == 1 and b[1] <= max_len],
        "tokenwise_seq_lens": [s for s in P.tokenwise_seq_lens if s <= max_len],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--num-pipeline-builder-workers", type=int, default=1)
    ap.add_argument("--num-compile-workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print the resolved bucket set and exit")
    a = ap.parse_args()

    import furiosa_llm
    from furiosa_llm.artifact import ArtifactBuilder, ArtifactConfig, BucketConfig, CompilerConfig, ModelConfig, ParallelConfig

    buckets = bucket_config_dict()
    rec = {
        "furiosa_llm_version": str(getattr(furiosa_llm, "__version__", None)),
        "model": a.model,
        "out": a.out,
        "tensor_parallel_size": a.tp,
        "max_model_len": MAX_MODEL_LEN,
        "buckets": buckets,
        "n_buckets": {k: len(v) for k, v in buckets.items()},
        "workers": {"pipeline_builder": a.num_pipeline_builder_workers, "compile": a.num_compile_workers},
    }
    print(json.dumps(rec, indent=1), flush=True)
    if a.dry_run:
        return 0
    os.environ.setdefault("RAY_COLOR_PREFIX", "1")
    os.environ.setdefault("RAY_DEDUP_LOGS_ALLOW_REGEX", "INFO:*[furiosa-llm]*")
    builder = ArtifactBuilder(
        a.model,
        a.name,
        model_config=ModelConfig(max_model_len=MAX_MODEL_LEN),
        parallel_config=ParallelConfig(tensor_parallel_size=a.tp, pipeline_parallel_size=1),
        bucket_config=BucketConfig(**{k: tuple(tuple(x) if isinstance(x, list) else x for x in v) for k, v in buckets.items()}),
        compiler_config=CompilerConfig(),
        artifact_config=ArtifactConfig(bundle_binaries=True),
    )
    t = time.time()
    builder.build(a.out, num_pipeline_builder_workers=a.num_pipeline_builder_workers,
                  num_compile_workers=a.num_compile_workers, cache_dir=a.cache_dir)
    rec["build_s"] = time.time() - t
    Path(a.out, "sdv_bmt_build_record.json").write_text(json.dumps(rec, indent=1))
    print("BUILD DONE", rec["build_s"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
