"""Plain-PyTorch re-definition of mmdetection3d CenterPoint-Pillars (pillar02, SECOND + SECONDFPN + CenterHead), tools only.

Used offline by ``export_onnx.py`` and the cross-checks; never imported by the benchmark core.  Written from the
official sources of mmdetection3d 1.4.0 (``voxel_encoders/pillar_encoder.py`` PillarFeatureNet with
``legacy=False``, ``voxel_encoders/utils.py`` PFNLayer, ``middle_encoders/pillar_scatter.py``, ``backbones/second.py``,
``necks/second_fpn.py``, ``dense_heads/centerpoint_head.py`` CenterHead/SeparateHead) and the checkpoint's config
``centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py``.  Module names equal the checkpoint keys,
so the state dict loads with ``strict=True``.

Input of ``forward``: the pillar tensors of the SDV-BMT payload, padded to 40,000 pillars:
``voxels`` float32 [P, 20, 5], ``num_points`` int32 [P] (0 for padding), ``coors`` int32 [P, 2] = (y, x).
Padded pillars (``num_points == 0``) are computed like the others (their features are discarded by the scatter,
which sends them to an extra canvas column that is cut off).  Output: the head tensors concatenated task by task
in the order reg(2), height(1), dim(3), rot(2), vel(2), heatmap(classes) -> [1, 70, 128, 128] float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

VOXEL = (0.2, 0.2, 8)
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
GRID = 512
TASKS = [["car"], ["truck", "construction_vehicle"], ["bus", "trailer"], ["barrier"], ["motorcycle", "bicycle"],
         ["pedestrian", "traffic_cone"]]
COMMON_HEADS = (("reg", 2), ("height", 1), ("dim", 3), ("rot", 2), ("vel", 2))
HEAD_CHANNELS = sum(c for _, c in COMMON_HEADS) * len(TASKS) + sum(len(t) for t in TASKS)  # 70


class PFNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(11, 64, bias=False)
        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        return torch.max(x, dim=1, keepdim=True)[0]


class PillarFeatureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pfn_layers = nn.ModuleList([PFNLayer()])
        self.vx, self.vy, self.vz = VOXEL
        self.x_offset = self.vx / 2 + PC_RANGE[0]
        self.y_offset = self.vy / 2 + PC_RANGE[1]
        self.z_offset = self.vz / 2 + PC_RANGE[2]

    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coors_zyx: torch.Tensor) -> torch.Tensor:
        dtype = features.dtype
        divisor = torch.clamp(num_points, min=1)  # padded pillars: avoid 0/0; their features are masked anyway
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / divisor.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coors_zyx[:, 2].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors_zyx[:, 1].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (coors_zyx[:, 0].to(dtype).unsqueeze(1) * self.vz + self.z_offset)
        features = torch.cat([features, f_cluster, f_center], dim=-1)
        mask = num_points.int().unsqueeze(1) > torch.arange(features.shape[1], dtype=torch.int).view(1, -1)
        features = features * mask.unsqueeze(-1).type_as(features)
        return self.pfn_layers[0](features).squeeze(1)


def scatter(pillar_features: torch.Tensor, num_points: torch.Tensor, coors_yx: torch.Tensor) -> torch.Tensor:
    """PointPillarsScatter; padded pillars go to an extra column that is cut off."""
    c = pillar_features.shape[1]
    idx = coors_yx[:, 0].long() * GRID + coors_yx[:, 1].long()
    idx = torch.where(num_points > 0, idx, torch.full_like(idx, GRID * GRID))
    canvas = torch.zeros(c, GRID * GRID + 1, dtype=pillar_features.dtype)
    canvas = canvas.index_put((torch.arange(c).unsqueeze(1), idx.unsqueeze(0)), pillar_features.t())
    return canvas[:, : GRID * GRID].reshape(1, c, GRID, GRID)


def _conv_bn_relu(cin: int, cout: int, k: int, s: int, p: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False), nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)]


class SECOND(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks, cin = [], 64
        for cout, n, s in ((64, 3, 2), (128, 5, 2), (256, 5, 2)):
            layers = _conv_bn_relu(cin, cout, 3, s, 1)
            for _ in range(n):
                layers += _conv_bn_relu(cout, cout, 3, 1, 1)
            blocks.append(nn.Sequential(*layers))
            cin = cout
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        return outs


class SECONDFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bn = lambda: nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)  # noqa: E731
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 128, 2, stride=2, bias=False), bn(), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(128, 128, 1, stride=1, bias=False), bn(), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False), bn(), nn.ReLU(inplace=True)),
        ])

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([d(x) for d, x in zip(self.deblocks, xs)], dim=1)


class ConvBN(nn.Module):
    """mmcv ConvModule(conv, BN, ReLU) with the checkpoint's attribute names."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        for name, c in COMMON_HEADS + (("heatmap", classes),):
            setattr(self, name, nn.Sequential(ConvBN(64, 64), nn.Conv2d(64, c, 3, padding=1, bias=True)))
        self.names = [n for n, _ in COMMON_HEADS] + ["heatmap"]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {n: getattr(self, n)(x) for n in self.names}


class CenterHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_conv = ConvBN(384, 64)
        self.task_heads = nn.ModuleList([SeparateHead(len(t)) for t in TASKS])

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        x = self.shared_conv(x)
        return [h(x) for h in self.task_heads]


class CenterPointPillars(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pts_voxel_encoder = PillarFeatureNet()
        self.pts_backbone = SECOND()
        self.pts_neck = SECONDFPN()
        self.pts_bbox_head = CenterHead()

    def heads(self, voxels: torch.Tensor, num_points: torch.Tensor, coors_yx: torch.Tensor):
        coors_zyx = torch.cat([torch.zeros_like(coors_yx[:, :1]), coors_yx], dim=1)
        feats = self.pts_voxel_encoder(voxels, num_points, coors_zyx)
        x = self.pts_neck(self.pts_backbone(scatter(feats, num_points, coors_yx)))
        return self.pts_bbox_head(x)

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coors_yx: torch.Tensor) -> torch.Tensor:
        return torch.cat([d[n] for d in self.heads(voxels, num_points, coors_yx)
                          for n in [h for h, _ in COMMON_HEADS] + ["heatmap"]], dim=1)


def load(checkpoint: str) -> CenterPointPillars:
    model = CenterPointPillars()
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model.eval()
