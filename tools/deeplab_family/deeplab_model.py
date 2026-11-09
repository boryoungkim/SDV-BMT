"""Plain-PyTorch re-definition of mmsegmentation DeepLabV3+ (ResNetV1c-D8 + DepthwiseSeparableASPPHead), tools only.

v0.1.2 DeepLab part.  Used offline by ``export_onnx.py`` (ONNX export on CPU) and by ``official_crosscheck.py``
(comparison with the official mmsegmentation implementation).  It is never imported by the benchmark core.

Written from the official sources at pinned commits (plan v03 section 3.7):
  mmsegmentation b040e147adfa027bbc071b624bedf0ae84dfc922: mmseg/models/backbones/resnet.py (ResNetV1c, BasicBlock,
    Bottleneck), mmseg/models/utils/res_layer.py, mmseg/models/decode_heads/{aspp_head,sep_aspp_head,decode_head}.py,
    mmseg/models/segmentors/encoder_decoder.py
  mmcv 57c4e25e06e2d4f8a9357c84bcd24089a284dc88 (v2.1.0): mmcv/cnn/bricks/{conv_module,depthwise_separable_conv_module}.py
and the training configs stored in each checkpoint's meta (``mmseg_version`` 0.9.0+640156c for R-18,
1.1.0+cee9004 for R-50/R-101).  Module names follow the official ones, so the checkpoint ``state_dict`` loads with
``strict=True`` (only ``auxiliary_head.*``, a training-only head, is left out).

Semantics reproduced (inference, ``test_cfg.mode='whole'``):
  * ResNetV1c: deep stem conv3x3(3->32, s2)+BN+ReLU, conv3x3(32->32)+BN+ReLU, conv3x3(32->64)+BN+ReLU, maxpool 3x3
    s2 p1; stages strides (1,2,1,1), dilations (1,1,2,4) with contract_dilation (first block of a dilated stage uses
    dilation // 2); style 'pytorch' (stride on the 3x3 conv); downsample = conv1x1(stride) + BN when the stride or
    the width changes.  BasicBlock: conv3x3(stride, dilation, padding=dilation) + BN + ReLU, conv3x3(padding=1, no
    dilation) + BN, + identity, ReLU.  Bottleneck: conv1x1 + BN + ReLU, conv3x3(stride, dilation, padding=dilation)
    + BN + ReLU, conv1x1 + BN, + identity, ReLU.
  * Head: image pool (global average -> ConvModule 1x1 -> bilinear resize to the feature size), ASPP 1x1 ConvModule
    and three DepthwiseSeparableConvModules (3x3, dilation = padding = 12/24/36; depthwise conv+BN+ReLU, pointwise
    conv+BN+ReLU), concat [pool, aspp...], bottleneck ConvModule 3x3, c1_bottleneck ConvModule 1x1 on stage 1,
    bilinear resize of the bottleneck output to the stage-1 size, concat [aspp, c1], two DepthwiseSeparableConvModules
    3x3, dropout (inference: identity), conv_seg 1x1 (with bias).  ConvModule = conv (no bias) + BN + ReLU.
  * Segmentor: logits resized to the input size (bilinear, align_corners=False), argmax over the 19 classes.
Every resize uses fixed output sizes derived from the static input size, so the ONNX graph holds constant sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 19
BN_EPS = 1e-5  # SyncBN default; the checkpoint configs set no eps


@dataclass(frozen=True)
class Variant:
    name: str
    depth: int
    block: str
    layers: tuple[int, int, int, int]
    head_in: int
    head_channels: int
    c1_in: int
    c1_channels: int


VARIANTS = {
    "r18": Variant("r18", 18, "basic", (2, 2, 2, 2), 512, 128, 64, 12),
    "r50": Variant("r50", 50, "bottleneck", (3, 4, 6, 3), 2048, 512, 256, 48),
    "r101": Variant("r101", 101, "bottleneck", (3, 4, 23, 3), 2048, 512, 256, 48),
}
STRIDES = (1, 2, 1, 1)
DILATIONS = (1, 1, 2, 4)
ASPP_DILATIONS = (1, 12, 24, 36)


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=BN_EPS)


class ConvModule(nn.Module):
    """mmcv ConvModule with norm: conv (bias=False) -> BN -> ReLU."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding, dilation, groups, bias=False)
        self.bn = _bn(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, padding: int, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, padding=padding, dilation=dilation, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int, dilation: int, downsample: nn.Module | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn1 = _bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = _bn(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int, dilation: int, downsample: nn.Module | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = _bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = _bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, 1, 0, bias=False)
        self.bn3 = _bn(planes * 4)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)


class ResNetV1c(nn.Module):
    def __init__(self, v: Variant) -> None:
        super().__init__()
        block = BasicBlock if v.block == "basic" else Bottleneck
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), _bn(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), _bn(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), _bn(64), nn.ReLU())
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        inplanes = 64
        for i, (n, stride, dilation) in enumerate(zip(v.layers, STRIDES, DILATIONS)):
            planes = 64 * 2 ** i
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                                           _bn(planes * block.expansion))
            first = dilation // 2 if dilation > 1 else dilation  # contract_dilation
            blocks = [block(inplanes, planes, stride, first, downsample)]
            inplanes = planes * block.expansion
            blocks += [block(inplanes, planes, 1, dilation, None) for _ in range(1, n)]
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.maxpool(self.stem(x))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class DepthwiseSeparableASPPHead(nn.Module):
    def __init__(self, v: Variant, feat_hw: tuple[int, int], c1_hw: tuple[int, int]) -> None:
        super().__init__()
        ch = v.head_channels
        self.feat_hw, self.c1_hw = feat_hw, c1_hw
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvModule(v.head_in, ch, 1))
        self.aspp_modules = nn.ModuleList(
            [ConvModule(v.head_in, ch, 1)]
            + [DepthwiseSeparableConvModule(v.head_in, ch, 3, padding=d, dilation=d) for d in ASPP_DILATIONS[1:]])
        self.bottleneck = ConvModule((len(ASPP_DILATIONS) + 1) * ch, ch, 3, padding=1)
        self.c1_bottleneck = ConvModule(v.c1_in, v.c1_channels, 1)
        self.sep_bottleneck = nn.Sequential(DepthwiseSeparableConvModule(ch + v.c1_channels, ch, 3, padding=1),
                                            DepthwiseSeparableConvModule(ch, ch, 3, padding=1))
        self.conv_seg = nn.Conv2d(ch, NUM_CLASSES, 1)

    def forward(self, c1: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        pool = F.interpolate(self.image_pool(c4), size=self.feat_hw, mode="bilinear", align_corners=False)
        x = torch.cat([pool] + [m(c4) for m in self.aspp_modules], dim=1)
        x = self.bottleneck(x)
        c1o = self.c1_bottleneck(c1)
        x = F.interpolate(x, size=self.c1_hw, mode="bilinear", align_corners=False)
        x = torch.cat([x, c1o], dim=1)
        return self.conv_seg(self.sep_bottleneck(x))


class DeepLabV3Plus(nn.Module):
    """``forward`` returns the uint8 label map (1x1xHxW); ``logits`` returns the 1/4-resolution class scores."""

    def __init__(self, variant: str, input_hw: tuple[int, int] = (1024, 2048)) -> None:
        super().__init__()
        v = VARIANTS[variant]
        h, w = input_hw
        if h % 8 or w % 8:
            raise ValueError("input size must be a multiple of 8")
        self.variant, self.input_hw = v, (h, w)
        self.backbone = ResNetV1c(v)
        self.decode_head = DepthwiseSeparableASPPHead(v, (h // 8, w // 8), (h // 4, w // 4))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        c1, _, _, c4 = self.backbone(x)
        return self.decode_head(c1, c4)

    def full_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.logits(x), size=self.input_hw, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.full_logits(x), dim=1, keepdim=True).to(torch.uint8)


def load_checkpoint(model: DeepLabV3Plus, path: str) -> dict:
    """Load an mmseg checkpoint strictly (``weights_only=True``; ``auxiliary_head.*`` left out).  Returns its meta."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in ck["state_dict"].items() if not k.startswith("auxiliary_head.")}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:  # pragma: no cover - strict=True raises first
        raise ValueError(f"state_dict mismatch: {result}")
    return ck.get("meta", {})


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fold an eval-mode BatchNorm into the preceding conv (float64 arithmetic, stored as float32)."""
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
    conv.weight = nn.Parameter((w * scale.view(-1, 1, 1, 1)).float())
    conv.bias = nn.Parameter(((b - bn.running_mean.detach().double()) * scale + bn.bias.detach().double()).float())


def fold_batchnorm(model: DeepLabV3Plus) -> int:
    """Fold every BatchNorm into its conv in place and replace it with Identity; returns the number folded."""
    n = 0
    stem = model.backbone.stem
    for ci, bi in ((0, 1), (3, 4), (6, 7)):
        _fuse(stem[ci], stem[bi])
        stem[bi] = nn.Identity()
        n += 1
    for module in list(model.modules()):
        if isinstance(module, ConvModule):
            _fuse(module.conv, module.bn)
            module.bn = nn.Identity()
            n += 1
        elif isinstance(module, (BasicBlock, Bottleneck)):
            pairs = (("conv1", "bn1"), ("conv2", "bn2")) + ((("conv3", "bn3"),) if isinstance(module, Bottleneck) else ())
            for c, b in pairs:
                _fuse(getattr(module, c), getattr(module, b))
                setattr(module, b, nn.Identity())
                n += 1
            if module.downsample is not None:
                _fuse(module.downsample[0], module.downsample[1])
                module.downsample[1] = nn.Identity()
                n += 1
    if any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):  # pragma: no cover - defensive
        raise RuntimeError("a BatchNorm was left unfolded")
    return n
