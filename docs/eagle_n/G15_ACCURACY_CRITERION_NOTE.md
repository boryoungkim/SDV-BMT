# G15 — Accuracy criterion (study note)

**Status: PROPOSED / DECISION PENDING.** The porting guide has not established
these thresholds as approved acceptance criteria. Boryoung must confirm them.

## Proposed rule

Compare Eagle-N mIoU with the GPU FP32 mIoU for the **same model**, using the
same Cityscapes validation inputs and evaluator. Define the absolute mIoU drop:

```text
drop = GPU_FP32_mIoU - Eagle_N_mIoU
```

The proposal is:

- **BF16:** pass the proposed quality threshold when `drop <= 0.01`.
- **8-bit** (`bfloat8_b` or `INT8`): pass the proposed quality threshold when
  `drop <= 0.02`.

These are absolute mIoU points, not relative percentage reductions.

## R-18 example

The guide lists R-18 GPU FP32 mIoU as `0.767566`:

- BF16 proposed minimum: `0.767566 - 0.01 = 0.757566`.
- 8-bit proposed minimum: `0.767566 - 0.02 = 0.747566`.

Use the measured Eagle-N mIoU to calculate the drop. For example, if BF16 gives
`0.760000`, the drop is `0.007566`, within the proposed `0.01` limit.

## How to report a miss

If the pipeline runs and produces an evaluation but exceeds the proposed drop,
report **“pipeline complete; proposed quality criterion not met.”** Do not label
the pipeline incomplete solely because of this quality result. Trying another
precision to meet the threshold requires Boryoung's decision.

## Source

`docs/porting/2026-11-10-cao-deeplab-eagle-n-porting-guide-v01.md`, section 4,
G15. The guide marks the criterion `[DECISION PENDING]`.
