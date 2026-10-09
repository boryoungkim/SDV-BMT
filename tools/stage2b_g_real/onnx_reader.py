"""Minimal ONNX reader (Python standard library only), Stage 2b-G real-model baseline.

Parses the protobuf wire format of an ONNX ``ModelProto`` far enough to read the graph: nodes with their
attributes, initializers (``raw_data`` or typed repeated fields), graph inputs/outputs with static shapes,
opset imports and producer.  Only what the offline converter needs is decoded; unknown fields are skipped.
No third-party package (``onnx``/``protobuf``) is used, so the converter adds no dependency.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Optional

# TensorProto.DataType values used here
FLOAT, UINT8, INT8, INT32, INT64, DOUBLE = 1, 2, 3, 6, 7, 11
_DTYPE_FORMAT = {FLOAT: ("f", 4), INT32: ("i", 4), INT64: ("q", 8), DOUBLE: ("d", 8), UINT8: ("B", 1), INT8: ("b", 1)}


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def fields(buf: bytes) -> list[tuple[int, int, Any]]:
    """(field number, wire type, value) for every field of one message, in wire order."""
    out = []
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            n, pos = _varint(buf, pos)
            value = buf[pos:pos + n]
            pos += n
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((number, wire, value))
    if pos != end:
        raise ValueError("truncated message")
    return out


def _packed_varints(value: Any, wire: int) -> list[int]:
    if wire == 0:
        return [_signed64(value)]
    vals, pos = [], 0
    while pos < len(value):
        v, pos = _varint(value, pos)
        vals.append(_signed64(v))
    return vals


def _packed_floats(value: Any, wire: int) -> list[float]:
    if wire == 5:
        return [struct.unpack("<f", value)[0]]
    return list(struct.unpack(f"<{len(value) // 4}f", value))


@dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    data_type: int
    raw: bytes  # little-endian element bytes

    def values(self) -> list[Any]:
        fmt, size = _DTYPE_FORMAT[self.data_type]
        return list(struct.unpack(f"<{len(self.raw) // size}{fmt}", self.raw))


@dataclass
class Node:
    op_type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    domain: str = ""


@dataclass
class ValueInfo:
    name: str
    elem_type: Optional[int]
    shape: Optional[tuple[Any, ...]]


@dataclass
class Model:
    ir_version: int
    producer_name: str
    producer_version: str
    opsets: dict[str, int]
    graph_name: str
    nodes: list[Node]
    initializers: dict[str, Tensor]
    inputs: list[ValueInfo]
    outputs: list[ValueInfo]
    value_info: list[ValueInfo]


def _tensor(buf: bytes) -> Tensor:
    dims: list[int] = []
    data_type, name, raw = 0, "", b""
    typed: dict[int, list[Any]] = {}
    for number, wire, value in fields(buf):
        if number == 1:
            dims.extend(_packed_varints(value, wire))
        elif number == 2:
            data_type = value
        elif number == 8:
            name = value.decode("utf-8")
        elif number == 9:
            raw = bytes(value)
        elif number == 4:
            typed.setdefault(FLOAT, []).extend(_packed_floats(value, wire))
        elif number in (5, 7):
            typed.setdefault(INT32 if number == 5 else INT64, []).extend(_packed_varints(value, wire))
        elif number == 14 and value == 1:
            raise ValueError(f"tensor {name}: external data is not supported")
    if not raw and data_type in typed:
        fmt, _ = _DTYPE_FORMAT[data_type]
        raw = struct.pack(f"<{len(typed[data_type])}{fmt}", *typed[data_type])
    return Tensor(name, tuple(dims), data_type, raw)


def _attribute(buf: bytes) -> tuple[str, Any]:
    name, kind = "", None
    f = i = s = t = None
    floats: list[float] = []
    ints: list[int] = []
    strings: list[bytes] = []
    for number, wire, value in fields(buf):
        if number == 1:
            name = value.decode("utf-8")
        elif number == 2:
            f = struct.unpack("<f", value)[0]
        elif number == 3:
            i = _signed64(value)
        elif number == 4:
            s = bytes(value)
        elif number == 5:
            t = _tensor(value)
        elif number == 7:
            floats.extend(_packed_floats(value, wire))
        elif number == 8:
            ints.extend(_packed_varints(value, wire))
        elif number == 9:
            strings.append(bytes(value))
        elif number == 20:
            kind = value
    # AttributeProto.AttributeType: FLOAT=1 INT=2 STRING=3 TENSOR=4 GRAPH=5 FLOATS=6 INTS=7 STRINGS=8
    by_kind = {1: f, 2: i, 3: s, 4: t, 6: floats, 7: ints, 8: strings}
    if kind in by_kind:
        return name, by_kind[kind]
    for candidate in (f, i, s, t):
        if candidate is not None:
            return name, candidate
    return name, ints or floats or strings


def _node(buf: bytes) -> Node:
    node = Node("", "", [], [])
    for number, _wire, value in fields(buf):
        if number == 1:
            node.inputs.append(value.decode("utf-8"))
        elif number == 2:
            node.outputs.append(value.decode("utf-8"))
        elif number == 3:
            node.name = value.decode("utf-8")
        elif number == 4:
            node.op_type = value.decode("utf-8")
        elif number == 5:
            key, val = _attribute(value)
            node.attrs[key] = val
        elif number == 7:
            node.domain = value.decode("utf-8")
    return node


def _value_info(buf: bytes) -> ValueInfo:
    name, elem, shape = "", None, None
    for number, _wire, value in fields(buf):
        if number == 1:
            name = value.decode("utf-8")
        elif number == 2:
            for n2, _w2, v2 in fields(value):
                if n2 != 1:  # tensor_type
                    continue
                for n3, _w3, v3 in fields(v2):
                    if n3 == 1:
                        elem = v3
                    elif n3 == 2:
                        dims: list[Any] = []
                        for n4, _w4, v4 in fields(v3):
                            if n4 != 1:
                                continue
                            dim: Any = None
                            for n5, _w5, v5 in fields(v4):
                                if n5 == 1:
                                    dim = _signed64(v5)
                                elif n5 == 2:
                                    dim = v5.decode("utf-8")
                            dims.append(dim)
                        shape = tuple(dims)
    return ValueInfo(name, elem, shape)


def load(path: str) -> Model:
    with open(path, "rb") as fh:
        data = fh.read()
    ir_version, producer, producer_version, graph_buf = 0, "", "", None
    opsets: dict[str, int] = {}
    for number, _wire, value in fields(data):
        if number == 1:
            ir_version = value
        elif number == 2:
            producer = value.decode("utf-8")
        elif number == 3:
            producer_version = value.decode("utf-8")
        elif number == 7:
            graph_buf = value
        elif number == 8:
            domain, version = "", 0
            for n2, _w2, v2 in fields(value):
                if n2 == 1:
                    domain = v2.decode("utf-8")
                elif n2 == 2:
                    version = v2
            opsets[domain] = version
    if graph_buf is None:
        raise ValueError(f"{path}: no graph")
    nodes, inits, inputs, outputs, infos, gname = [], {}, [], [], [], ""
    for number, _wire, value in fields(graph_buf):
        if number == 1:
            nodes.append(_node(value))
        elif number == 2:
            gname = value.decode("utf-8")
        elif number == 5:
            t = _tensor(value)
            inits[t.name] = t
        elif number == 11:
            inputs.append(_value_info(value))
        elif number == 12:
            outputs.append(_value_info(value))
        elif number == 13:
            infos.append(_value_info(value))
    return Model(ir_version, producer, producer_version, opsets, gname, nodes, inits, inputs, outputs, infos)
