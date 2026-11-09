"""Plan-level conv-condition check of the DeepLab numeric gate (plan v03 section 8.1 step 5; stdlib only).

The canonical plan and the diagnostic (logits) plan must share every convolution exactly: the same cuDNN descriptor
values, weight/bias offsets, activation and residual, the same weights, arena and input, and the same steps; they
differ only in the last convolution's destination (the output buffer in the diagnostic plan) and the canonical
plan's final resize + argmax.  The device part (identical realized algorithm tables and heuristic choices) is
checked by validate_numerics.py from the native prepare facts.
"""

from __future__ import annotations

CONV_KEYS = ("x", "residual", "res_before_act", "act", "w_offset", "b_offset", "c", "h", "w", "k", "r", "s", "oh",
             "ow", "pad_h", "pad_w", "stride_h", "stride_w", "dil_h", "dil_w", "group")


def conv_conditions(canon: dict, diag: dict) -> dict:
    """Plan-level part of the conv-condition check (the device part compares the realized tables)."""
    problems = []
    for key in ("schema", "model", "weights", "arena_bytes", "input"):
        if canon[key] != diag[key]:
            problems.append(f"{key} differs")
    cs, ds = canon["steps"], diag["steps"]
    if [s["op"] for s in cs[:-2]] != [s["op"] for s in ds] or [s["op"] for s in cs[-2:]] != ["resize_bilinear", "argmax_u8"]:
        problems.append("step sequence differs beyond the final resize and argmax")
    convs_c = [s for s in cs if s["op"] == "conv"]
    convs_d = [s for s in ds if s["op"] == "conv"]
    if len(convs_c) != len(convs_d):
        problems.append(f"convolution count {len(convs_c)} vs {len(convs_d)}")
    for i, (a, b) in enumerate(zip(convs_c, convs_d)):
        for key in CONV_KEYS:
            if a.get(key) != b.get(key):
                problems.append(f"conv {i}: {key} differs")
        last = i == len(convs_c) - 1
        if not last and a["y"] != b["y"]:
            problems.append(f"conv {i}: destination differs")
        if last and b["y"] != {"buf": "output", "offset": 0}:
            problems.append("diagnostic plan: the last convolution must write the output buffer")
    for i, (a, b) in enumerate(zip(cs[:-2], ds)):
        if a["op"] != "conv" and a != b:
            problems.append(f"step {i} ({a['op']}) differs")
    return {"convolutions": len(convs_c), "plan_problems": problems}
