# Section 5.4 — Data formats (study note)

## Main point

SDV-BMT's public input contract and the Eagle-N runtime's internal tensor
format are separate layers. The SDV-BMT input remains canonical RGB, NCHW,
FP32. The adapter/runtime may convert layout or precision for TTNN, but must
preserve tensor meaning and declare where each conversion occurs.

## Precision

- TT-Forge-ONNX exposes data-format configuration; upstream benchmarks commonly
  use BF16.
- BOS documentation mentions BF16 for TT-XLA. This does not prove every BOS
  compiler path applies the same conversion automatically.
- A workload named `bf16` should mean BF16 throughout the model. If layers use
  different formats, declare it as mixed precision and record the per-layer
  formats.
- Verify the behavior with the installed BOS SDK and compare numerical/task
  quality against the agreed reference.

## Layout and convolution

- SDV-BMT canonical input: NCHW, for example `[1, 3, 1024, 2048]`.
- The guide reports `ttnn.conv2d` expects NHWC. Conversion to `[1, 1024, 2048,
  3]` may therefore be required before convolution.
- A transpose/permute changes how values are indexed in memory; using the wrong
  dimension order can produce incorrect output even if the operation runs.
- The guide says `bfloat8_b` requires TILE layout. Weight padding and NCHW to
  NHWC conversion may occur on the host.

## Measurement implications

Record whether layout conversion, padding, and dtype conversion happen during
input staging or inside the measured submit/execute path. These steps use time
and memory, and changing their placement changes the latency boundary. Do not
assume a host-side conversion is free or included/excluded without checking the
adapter implementation and measurement declaration.

## Verification status

The format/layout statements above summarize the porting guide's upstream
references. They are not proof that the installed BOS Eagle-N SDK supports each
combination. Verify the selected precision, layout, padding, correctness, and
memory behavior on the actual SDK/device before accepting measurements.

## Source

`docs/porting/2026-11-10-cao-deeplab-eagle-n-porting-guide-v01.md`, section 5.4.
