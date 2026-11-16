"""Export CenterPoint-Pillars (mmdet3d pillar02 circle-NMS checkpoint) to ONNX with BatchNorm folded (offline, tools only).

Run with the system Python (torch CPU and onnx are used by this offline tool only; the benchmark core does not use them).

  1. load the checkpoint into the plain-PyTorch re-definition (``centerpoint_model.py``, ``strict=True``);
  2. fold every BatchNorm into the preceding linear / convolution / transposed convolution (float64, stored float32);
  3. export the folded model (inputs: the SDV-BMT pillar payload tensors ``voxels`` float32 [40000, 20, 5],
     ``num_points`` int32 [40000], ``coors`` int32 [40000, 2] = (y, x); output ``heads`` float32 [1, 70, 128, 128])
     with torch.onnx.export (TorchScript exporter, opset 17, constant folding);
  4. export gate: the folded PyTorch model vs ONNX Runtime CPU on the first ``--count`` payloads of an SDV-BMT index,
     ``max|a-b| / max|b|`` per sample <= 1e-4; the unfolded re-definition vs the folded model is reported too.

    python3 tools/centerpoint/export_onnx.py --checkpoint CKPT --index INDEX --out MODEL.onnx --report REPORT.json [--count 8]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import centerpoint_model as M  # noqa: E402

H = struct.Struct("<8s8I")


def payload_tensors(path: Path):
    b = path.read_bytes()
    _, P, MP, MPTS, F, *_ = H.unpack_from(b)
    o = H.size
    v = np.frombuffer(b, "<f4", MP * MPTS * F, o).reshape(MP, MPTS, F)
    o += v.nbytes
    n = np.frombuffer(b, "<i4", MP, o)
    o += n.nbytes
    yx = np.frombuffer(b, "<i4", MP * 2, o).reshape(MP, 2)
    return v.copy(), n.copy(), yx.copy()


def bn_scale_shift(bn) -> tuple[torch.Tensor, torch.Tensor]:
    s = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
    return s, bn.bias.double() - bn.running_mean.double() * s


def fold_conv(conv: nn.Conv2d, bn) -> nn.Conv2d:
    s, t = bn_scale_shift(bn)
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups, bias=True)
    out.weight.data = (conv.weight.double() * s.view(-1, 1, 1, 1)).float()
    b0 = conv.bias.double() if conv.bias is not None else torch.zeros_like(s)
    out.bias.data = (b0 * s + t).float()
    return out


def fold_deconv(conv: nn.ConvTranspose2d, bn) -> nn.ConvTranspose2d:
    s, t = bn_scale_shift(bn)
    out = nn.ConvTranspose2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, bias=True)
    out.weight.data = (conv.weight.double() * s.view(1, -1, 1, 1)).float()
    out.bias.data = t.float()
    return out


class FoldedPFN(nn.Module):
    def __init__(self, pfn: M.PillarFeatureNet) -> None:
        super().__init__()
        layer = pfn.pfn_layers[0]
        s, t = bn_scale_shift(layer.norm)
        self.linear = nn.Linear(11, 64, bias=True)
        self.linear.weight.data = (layer.linear.weight.double() * s.view(-1, 1)).float()
        self.linear.bias.data = t.float()
        self.src = pfn

    def forward(self, features, num_points, coors_zyx):
        p = self.src
        dtype = features.dtype
        divisor = torch.clamp(num_points, min=1)
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / divisor.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coors_zyx[:, 2].to(dtype).unsqueeze(1) * p.vx + p.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors_zyx[:, 1].to(dtype).unsqueeze(1) * p.vy + p.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (coors_zyx[:, 0].to(dtype).unsqueeze(1) * p.vz + p.z_offset)
        x = torch.cat([features, f_cluster, f_center], dim=-1)
        mask = num_points.int().unsqueeze(1) > torch.arange(x.shape[1], dtype=torch.int).view(1, -1)
        x = x * mask.unsqueeze(-1).type_as(x)
        x = torch.relu(self.linear(x))
        return torch.max(x, dim=1)[0]


class Folded(nn.Module):
    """The re-definition with every BatchNorm folded (same graph otherwise)."""

    def __init__(self, m: M.CenterPointPillars) -> None:
        super().__init__()
        self.pfn = FoldedPFN(m.pts_voxel_encoder)
        blocks = []
        for b in m.pts_backbone.blocks:
            mods, layers = list(b), []
            for i in range(0, len(mods), 3):
                layers += [fold_conv(mods[i], mods[i + 1]), nn.ReLU(inplace=True)]
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        d = m.pts_neck.deblocks
        self.deblocks = nn.ModuleList([
            nn.Sequential(fold_conv(d[0][0], d[0][1]), nn.ReLU(inplace=True)),
            nn.Sequential(fold_conv(d[1][0], d[1][1]), nn.ReLU(inplace=True)),
            nn.Sequential(fold_deconv(d[2][0], d[2][1]), nn.ReLU(inplace=True))])
        h = m.pts_bbox_head
        self.shared = nn.Sequential(fold_conv(h.shared_conv.conv, h.shared_conv.bn), nn.ReLU(inplace=True))
        self.heads = nn.ModuleList()
        for task in h.task_heads:
            for name in task.names:
                seq = getattr(task, name)
                self.heads.append(nn.Sequential(fold_conv(seq[0].conv, seq[0].bn), nn.ReLU(inplace=True), seq[1]))

    def forward(self, voxels, num_points, coors):
        coors_zyx = torch.cat([torch.zeros_like(coors[:, :1]), coors], dim=1)
        x = M.scatter(self.pfn(voxels, num_points, coors_zyx), num_points, coors)
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        x = torch.cat([d(o) for d, o in zip(self.deblocks, outs)], dim=1)
        x = self.shared(x)
        return torch.cat([h(x) for h in self.heads], dim=1)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(b).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = M.load(a.checkpoint)
    folded = Folded(copy.deepcopy(model)).eval()
    rows = [json.loads(l) for l in open(a.index)][: a.count]
    idx_dir = Path(a.index).parent
    v, n, yx = payload_tensors(idx_dir / rows[0]["payload_file"])
    args = (torch.from_numpy(v), torch.from_numpy(n), torch.from_numpy(yx))
    torch.onnx.export(folded, args, a.out, opset_version=17, do_constant_folding=True,
                      input_names=["voxels", "num_points", "coors"], output_names=["heads"], dynamo=False)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.out, so, providers=["CPUExecutionProvider"])
    per = []
    for r in rows:
        v, n, yx = payload_tensors(idx_dir / r["payload_file"])
        with torch.no_grad():
            ref = folded(torch.from_numpy(v), torch.from_numpy(n), torch.from_numpy(yx)).numpy()
            unf = model(torch.from_numpy(v), torch.from_numpy(n), torch.from_numpy(yx)).numpy()
        got = sess.run(["heads"], {"voxels": v, "num_points": n, "coors": yx})[0]
        per.append({"token": r["source_item_id"], "ort_vs_folded": rel(got, ref), "folded_vs_unfolded": rel(ref, unf)})
        print(per[-1])
    onnx_bytes = Path(a.out).read_bytes()
    report = {"criterion": "export gate: max|ORT - folded torch| / max|folded torch| <= 1e-4 per sample",
              "worst_ort_vs_folded": max(p["ort_vs_folded"] for p in per),
              "worst_folded_vs_unfolded": max(p["folded_vs_unfolded"] for p in per),
              "pass": max(p["ort_vs_folded"] for p in per) <= 1e-4, "samples": per,
              "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(), "onnx_bytes": len(onnx_bytes),
              "torch": torch.__version__, "onnxruntime": ort.__version__,
              "export_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest()}
    Path(a.report).write_text(json.dumps(report, indent=1))
    print("worst ort_vs_folded", report["worst_ort_vs_folded"], "pass", report["pass"], "sha", report["onnx_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
