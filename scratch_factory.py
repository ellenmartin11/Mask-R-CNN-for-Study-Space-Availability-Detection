"""
scratch_factory.py
------------------
Defines ScratchNet — a fully custom backbone for Mask R-CNN.

Architecture (reverse-engineered from scratch_best_weights_50ep.pth):
  Stem  : Conv(3→32, 3×3, stride=2) + BN + ReLU                    [stride 2]
  Stage1: DualPathBlock(32→64,  s=2), DualPathBlock(64→64,  s=1)    [stride 4  → C2]
  Stage2: DualPathBlock(64→128, s=2), ×2 DualPathBlock(128→128)     [stride 8  → C3]
  Stage3: DualPathBlock(128→192,s=2), ×3 DualPathBlock(192→192)     [stride 16 → C4]
  Stage4: DualPathBlock(192→288,s=2), DualPathBlock(288→288, s=1)   [stride 32 → C5]
  FPN   : in=[64,128,192,288], out_channels=96, 5 levels (P2–P6)

Each DualPathBlock:
  pre  = BN(x)
  a    = Conv(in, out//2, 3×3, stride=s)               [path_a]
  b    = DepthwiseConv(in, in, 3×3, s) → Conv(in, out//2, 1×1) [path_b_dw / path_b_pw]
  merged = merge_proj( BN( cat(a, b) ) )               [merge_bn / merge_proj]
  out  = ReLU( merged + shortcut(x) )

Shortcut: Sequential( MaxPool2d(s) , Conv(in→out, 1×1) )  when dims differ
          Identity                                          otherwise

Total trainable parameters: ~4.2 M  (≈ 1/10 of ResNet-50 Mask R-CNN at ~44 M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DualPathBlock(nn.Module):
    """
    Custom pre-activation dual-path residual block.

    Two parallel feature paths are computed from the pre-normalised input,
    concatenated, projected with a 1×1 conv, and added back to the shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        half = out_ch // 2

        self.bn_pre    = nn.BatchNorm2d(in_ch)
        self.path_a    = nn.Conv2d(in_ch, half, 3, stride=stride, padding=1, bias=False)
        self.path_b_dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                   groups=in_ch, bias=False)
        self.path_b_pw = nn.Conv2d(in_ch, half, 1, bias=False)
        self.merge_bn  = nn.BatchNorm2d(out_ch)
        self.merge_proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride, ceil_mode=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre    = self.bn_pre(x)
        a      = self.path_a(pre)
        b      = self.path_b_pw(self.path_b_dw(pre))
        merged = self.merge_proj(self.merge_bn(torch.cat([a, b], dim=1)))
        return F.relu(merged + self.shortcut(x), inplace=True)


# ---------------------------------------------------------------------------
# ScratchNet backbone body
# ---------------------------------------------------------------------------

class ScratchNet(nn.Module):
    """
    Custom CNN backbone — no weights borrowed from any known architecture.

    Exposes four feature maps (C2–C5) via forward(), keyed '0'–'3' so that
    BackboneWithFPN can attach FPN lateral connections directly.
    """

    def __init__(self):
        super().__init__()

        # stride=2  →  32-ch feature map
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # C2 – stride=4,  out=64
        self.stage1 = nn.Sequential(
            DualPathBlock(32,  64, stride=2),
            DualPathBlock(64,  64, stride=1),
        )

        # C3 – stride=8,  out=128
        self.stage2 = nn.Sequential(
            DualPathBlock(64,  128, stride=2),
            DualPathBlock(128, 128, stride=1),
            DualPathBlock(128, 128, stride=1),
        )

        # C4 – stride=16, out=192
        self.stage3 = nn.Sequential(
            DualPathBlock(128, 192, stride=2),
            DualPathBlock(192, 192, stride=1),
            DualPathBlock(192, 192, stride=1),
            DualPathBlock(192, 192, stride=1),
        )

        # C5 – stride=32, out=288
        self.stage4 = nn.Sequential(
            DualPathBlock(192, 288, stride=2),
            DualPathBlock(288, 288, stride=1),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        x  = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return OrderedDict([('0', c2), ('1', c3), ('2', c4), ('3', c5)])


# ---------------------------------------------------------------------------
# Full Mask R-CNN construction
# ---------------------------------------------------------------------------

def build_scratch_mask_rcnn(num_classes: int = 9) -> MaskRCNN:
    """
    Assembles Mask R-CNN around ScratchNet.

    Args:
        num_classes: foreground classes + background (default 9 = 8 UNH classes + bg)

    Returns:
        Untrained MaskRCNN model (~4.2 M parameters).
    """
    body = ScratchNet()

    backbone = BackboneWithFPN(
        backbone=body,
        return_layers={'stage1': '0', 'stage2': '1', 'stage3': '2', 'stage4': '3'},
        in_channels_list=[64, 128, 192, 288],
        out_channels=96,
        extra_blocks=LastLevelMaxPool(),
    )

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2,
    )

    # 96 * 7 * 7 = 4704 → 256 → 256
    box_head      = TwoMLPHead(in_channels=96 * 7 * 7, representation_size=256)
    box_predictor = FastRCNNPredictor(in_channels=256, num_classes=num_classes)

    # 4 × Conv(96→96, 3×3)
    mask_head      = MaskRCNNHeads(in_channels=96, layers=(96, 96, 96, 96), dilation=1)
    # ConvTranspose(96→48, 2×2) + Conv(48→9, 1×1)
    mask_predictor = MaskRCNNPredictor(in_channels=96, dim_reduced=48,
                                       num_classes=num_classes)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=None,           # heads are provided explicitly below
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=box_predictor,
        mask_roi_pool=mask_roi_pool,
        mask_head=mask_head,
        mask_predictor=mask_predictor,
    )

    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}  (~{total/1e6:.2f} M)")
    return total


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_scratch_mask_rcnn(num_classes=9)
    count_parameters(model)

    model.eval()
    dummy = [torch.randn(3, 800, 1067)]
    with torch.no_grad():
        out = model(dummy)
    print("Output keys:", list(out[0].keys()))
    print("boxes shape:", out[0]['boxes'].shape)
    print("masks shape:", out[0]['masks'].shape)
