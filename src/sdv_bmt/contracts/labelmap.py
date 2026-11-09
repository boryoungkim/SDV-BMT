"""Label-map codec ``labelmap-u8-rowrle/1`` (v0.1.2 DeepLab part): lossless row-wise run-length coding of a
uint8 class-index map, used for the segmentation task outputs and for the frozen ground truth.

Layout (little-endian):
  header, 16 bytes: magic ``SDVLMRL1`` (8 bytes), item index u32 (``NO_ITEM`` = 0xFFFFFFFF for a ground truth),
                    width u16, height u16
  one record per row, top to bottom:
    ``0x01, runs u16, runs x (value u8, length u16)`` when that is shorter than the raw row, else
    ``0x00, width raw bytes``.
  Runs never span rows; each run length is in 1..width; the lengths of a row sum to width.  The encoder picks the
  run-length record only when ``3 + 3*runs < 1 + width``, so the output is unique for a given map and never larger
  than ``16 + height * (1 + width)`` bytes.

The native measurement plane has a byte-identical twin (``native/src/common/labelmap_output.hpp``); it lives in ``contracts`` because both the workload (encode) and the
evaluator (decode) use it.
"""

from __future__ import annotations

import re
import struct

CODEC_ID = "labelmap-u8-rowrle/1"
MAGIC = b"SDVLMRL1"
HEADER = struct.Struct("<8sIHH")
NO_ITEM = 0xFFFFFFFF
_RUN = re.compile(rb"(.)\1*", re.S)


class CodecError(ValueError):
    pass


def max_encoded_bytes(width: int, height: int) -> int:
    return HEADER.size + height * (1 + width)


def encode(labels: bytes, width: int, height: int, item_index: int = NO_ITEM) -> bytes:
    if not (0 < width <= 0xFFFF and 0 < height <= 0xFFFF) or len(labels) != width * height:
        raise CodecError(f"label map must be {width}x{height} bytes")
    if not 0 <= item_index <= 0xFFFFFFFF:
        raise CodecError("item index out of range")
    out = bytearray(HEADER.pack(MAGIC, item_index, width, height))
    for y in range(height):
        row = labels[y * width:(y + 1) * width]
        runs = [(m.start(), m.end()) for m in _RUN.finditer(row)]
        if 3 + 3 * len(runs) < 1 + width:
            out += struct.pack("<BH", 1, len(runs))
            for s, e in runs:
                out += struct.pack("<BH", row[s], e - s)
        else:
            out.append(0)
            out += row
    return bytes(out)


def decode(blob: bytes) -> tuple[int, int, int, bytes]:
    """(item index, width, height, label bytes) of an encoded map; raises CodecError on any malformed input."""
    if len(blob) < HEADER.size:
        raise CodecError("label map blob shorter than its header")
    magic, item, width, height = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or width == 0 or height == 0:
        raise CodecError("not a labelmap-u8-rowrle/1 blob")
    out = bytearray()
    pos = HEADER.size
    for _ in range(height):
        if pos >= len(blob):
            raise CodecError("label map blob truncated")
        mode = blob[pos]
        pos += 1
        if mode == 0:
            row = blob[pos:pos + width]
            if len(row) != width:
                raise CodecError("label map blob truncated")
            out += row
            pos += width
        elif mode == 1:
            if pos + 2 > len(blob):
                raise CodecError("label map blob truncated")
            (n,) = struct.unpack_from("<H", blob, pos)
            pos += 2
            if pos + 3 * n > len(blob) or 3 + 3 * n >= 1 + width:
                raise CodecError("label map row record is invalid")
            total = 0
            prev = -1
            for k in range(n):
                v, length = struct.unpack_from("<BH", blob, pos + 3 * k)
                if length == 0 or v == prev:
                    raise CodecError("label map run is empty or not maximal")
                out += bytes((v,)) * length
                total += length
                prev = v
            if total != width:
                raise CodecError("label map row runs do not sum to the width")
            pos += 3 * n
        else:
            raise CodecError(f"unknown label map row mode {mode}")
    if pos != len(blob):
        raise CodecError("trailing bytes after the label map")
    return item, width, height, bytes(out)
