"""Model inputs for the DeepLab offline tools (tools only): the canonical prepared tensor of each input image.

The tensor bytes come from the workload's own normalization (``sdv_bmt.workload.deeplab.normalize_rgb``), so the
tools feed ONNX Runtime, PyTorch and the GPU exactly what a run stages.

* ``synthetic``: deterministic image-like RGB8 images (smooth fields, blocks, edges and noise from a seeded
  numpy generator), used until the Cityscapes val images are available; they exercise the arithmetic, not the
  task.
* a directory of PPM files: the first ``count`` files in sorted name order.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from sdv_bmt.workload.deeplab import normalize_rgb  # noqa: E402
from sdv_bmt.workload.yolox import parse_ppm  # noqa: E402


def synthetic_rgb(index: int, h: int, w: int) -> bytes:
    rng = np.random.default_rng(20261106 + index)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))
    for c in range(3):
        fx, fy, ph = rng.uniform(0.5, 6.0), rng.uniform(0.5, 6.0), rng.uniform(0, 2 * np.pi)
        img[..., c] = 128 + 90 * np.sin(2 * np.pi * (fx * xx / w + fy * yy / h) + ph)
    for _ in range(12):  # flat blocks with sharp edges
        y0, x0 = rng.integers(0, h - 16), rng.integers(0, w - 16)
        img[y0:y0 + rng.integers(16, h // 3), x0:x0 + rng.integers(16, w // 3)] = rng.uniform(0, 255, 3)
    img += rng.normal(0, 8, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8).tobytes()


def tensor_from_rgb(rgb: bytes, h: int, w: int) -> np.ndarray:
    return np.frombuffer(normalize_rgb(rgb, w, h), dtype="<f4").reshape(1, 3, h, w).copy()


def iter_inputs(spec: str, count: int, h: int, w: int) -> Iterator[tuple[str, np.ndarray]]:
    if spec == "synthetic":
        for i in range(count):
            yield f"synthetic-{i}", tensor_from_rgb(synthetic_rgb(i, h, w), h, w)
        return
    files = sorted(Path(spec).glob("*.ppm"))[:count]
    if len(files) < count:
        raise SystemExit(f"{spec}: fewer than {count} PPM files")
    for f in files:
        pw, ph, rgb = parse_ppm(f.read_bytes())
        if (pw, ph) != (w, h):
            raise SystemExit(f"{f}: {pw}x{ph}, expected {w}x{h}")
        yield f.stem, tensor_from_rgb(rgb, h, w)
