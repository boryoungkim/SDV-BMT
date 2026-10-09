"""Independent integer CPU implementation of the SDV-BMT reference conformance program v1.

It mirrors ``sdv_ref_convnet_v1.cu`` with exact integer arithmetic (values in
units of 1/128).  Because every intermediate value of the device program is
exactly representable in float32, the device logits must equal these integers
bit for bit.  Used to produce the golden output reference before any GPU run.
"""

from __future__ import annotations

SIDE = 128


def w1(o: int, c: int, ky: int, kx: int) -> int:
    return ((o * 131 + c * 31 + ky * 7 + kx * 3) % 7) - 3


def b1(o: int) -> int:
    return (o % 5) - 2


def w2(o: int, c: int, ky: int, kx: int) -> int:
    return ((o * 17 + c * 13 + ky * 5 + kx * 11) % 7) - 3


def b2(o: int) -> int:
    return (o % 3) - 1


def wf(j: int, o: int) -> int:
    return ((j * 29 + o * 23) % 9) - 4


def bf(j: int) -> int:
    return j - 5


def _conv(planes: list[list[list[int]]], cout: int, wfn, bfn) -> list[list[list[int]]]:
    """3x3 convolution with zero padding, integer bias, ReLU (numerators in 1/128 units)."""
    shifted = [{-1: [[0] + row[:-1] for row in p], 0: p, 1: [row[1:] + [0] for row in p]} for p in planes]
    out = []
    for o in range(cout):
        acc = [[bfn(o)] * SIDE for _ in range(SIDE)]
        for c, sc in enumerate(shifted):
            for ky in range(3):
                dy = ky - 1
                for kx in range(3):
                    w = wfn(o, c, ky, kx)
                    if w == 0:
                        continue
                    rows = sc[kx - 1]
                    for y in range(SIDE):
                        yy = y + dy
                        if 0 <= yy < SIDE:
                            acc[y] = [a + w * s for a, s in zip(acc[y], rows[yy])]
        out.append([[v if v > 0 else 0 for v in row] for row in acc])
    return out


def reference_logits(image: bytes) -> list[int]:
    """Logits of one 128x128 RGB uint8 image (HWC).  The device program sees the
    x4 nearest upsample, whose 4x4 average pool is exactly this image."""
    if len(image) != SIDE * SIDE * 3:
        raise ValueError("image must be 128x128x3 bytes")
    planes = [[[image[(y * SIDE + x) * 3 + c] - 128 for x in range(SIDE)] for y in range(SIDE)] for c in range(3)]
    f1 = _conv(planes, 16, w1, b1)
    f2 = _conv(f1, 16, w2, b2)
    gap = [sum(sum(row) for row in plane) for plane in f2]
    return [bf(j) + sum(gap[o] * wf(j, o) for o in range(16)) for j in range(10)]
