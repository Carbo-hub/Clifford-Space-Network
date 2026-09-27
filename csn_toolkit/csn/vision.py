"""
Computer Vision Architecture powered by Clifford Space Networks (CSN).

Components:
1. SpatialPatchStem: Hierarchical convolutional downsampling stem.
2. CliffordSpatialRotorBlockCl4: Spin(4) Rotor Block with 2D Bidirectional Spatial Scan.
3. CliffordSpatialRotorBlockCl8: Spin(8) Rotor Block with 2D Bidirectional Spatial Scan.
4. CliffordFPN: Feature Pyramid Network for multi-scale feature aggregation (P3, P4, P5).
5. CliffordDetectionHead: Decoupled multi-scale heads for heatmaps, bounding boxes, and subpixel offsets.
6. CliffordDetModel: Full End-to-End Object Detector / Backbone.
7. ModelEMA: Exponential Moving Average with dynamic warmup for weights and buffers.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn.rotors import cayley_rotor_4x4, cayley_rotor_16x16
from csn.scan import parallel_rotor_scan


class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpatialPatchStem(nn.Module):
    """Hierarchical downsampling stem outputting C3 (stride 8), C4 (stride 16), C5 (stride 32)."""
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.p1 = ConvBNAct(in_channels, base_channels // 2, k=3, s=2, p=1)      # Stride 2 (320x320)
        self.p2 = ConvBNAct(base_channels // 2, base_channels, k=3, s=2, p=1)     # Stride 4 (160x160)
        self.p3_conv = ConvBNAct(base_channels, base_channels * 2, k=3, s=2, p=1) # Stride 8 (80x80)
        self.p4_conv = ConvBNAct(base_channels * 2, base_channels * 4, k=3, s=2, p=1) # Stride 16 (40x40)
        self.p5_conv = ConvBNAct(base_channels * 4, base_channels * 4, k=3, s=2, p=1) # Stride 32 (20x20)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.p1(x)
        x2 = self.p2(x1)
        c3 = self.p3_conv(x2)
        c4 = self.p4_conv(c3)
        c5 = self.p5_conv(c4)
        return c3, c4, c5


class CliffordSpatialRotorBlockCl4(nn.Module):
    """2D Bidirectional Spatial Scan using Spin(4) Lie Rotors in Cl(4)."""
    def __init__(self, in_channels: int, num_neurons: int = 2, use_schulz: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_neurons = num_neurons
        self.use_schulz = use_schulz

        self.in_proj = nn.Conv2d(in_channels, num_neurons * 16, kernel_size=1, bias=False)
        self.biv_proj = nn.Conv2d(in_channels, num_neurons * 6, kernel_size=1, bias=False)
        self.metric_proj = nn.Conv2d(in_channels, num_neurons, kernel_size=3, padding=1, bias=False)
        self.gamma_log = nn.Parameter(torch.zeros(num_neurons))
        self.lambda_sw = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Conv2d(num_neurons * 16, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=True)

        triu_idx = torch.triu_indices(4, 4, offset=1)
        self.register_buffer("triu_u", triu_idx[0])
        self.register_buffer("triu_v", triu_idx[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ds = F.softplus(self.metric_proj(x)).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons)
        C_inj = self.in_proj(x).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons, 4, 4)

        biv = self.biv_proj(x).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons, 6)
        skew = torch.zeros(B, H * W, self.num_neurons, 4, 4, dtype=biv.dtype, device=x.device)
        skew[..., self.triu_u, self.triu_v] = biv
        skew[..., self.triu_v, self.triu_u] = -biv

        J_sw = 0.5 * (C_inj - C_inj.transpose(-1, -2))
        Omega = (skew + torch.tanh(self.lambda_sw) * J_sw) * (0.5 * ds.unsqueeze(-1).unsqueeze(-1))
        Omega = Omega.clamp(min=-3.0, max=3.0)

        Q = cayley_rotor_4x4(Omega, use_schulz=self.use_schulz)
        gamma = F.softplus(self.gamma_log).view(1, 1, self.num_neurons, 1, 1)
        decay = torch.sigmoid(-gamma * ds.unsqueeze(-1).unsqueeze(-1))
        M = decay * Q

        # Horizontal 2D Scan
        M_h = M.view(B * H, W, self.num_neurons, 4, 4)
        C_h = C_inj.view(B * H, W, self.num_neurons, 4, 4)
        psi_h = (parallel_rotor_scan(M_h, C_h, is_spinor_bundle=True) +
                 parallel_rotor_scan(M_h.flip(1), C_h.flip(1), is_spinor_bundle=True).flip(1)) * 0.5

        # Vertical 2D Scan
        M_v = M.view(B, H, W, self.num_neurons, 4, 4).permute(0, 2, 1, 3, 4, 5).contiguous().view(B * W, H, self.num_neurons, 4, 4)
        C_v = psi_h.view(B, H, W, self.num_neurons, 4, 4).permute(0, 2, 1, 3, 4, 5).contiguous().view(B * W, H, self.num_neurons, 4, 4)
        psi_v = (parallel_rotor_scan(M_v, C_v, is_spinor_bundle=True) +
                 parallel_rotor_scan(M_v.flip(1), C_v.flip(1), is_spinor_bundle=True).flip(1)) * 0.5

        out = psi_v.view(B, W, H, self.num_neurons, 4, 4).permute(0, 2, 1, 3, 4, 5).contiguous()
        out = out.view(B, H, W, self.num_neurons * 16).permute(0, 3, 1, 2).contiguous()
        return self.act(self.norm(x + self.out_proj(out)))


class CliffordSpatialRotorBlockCl8(nn.Module):
    """2D Bidirectional Spatial Scan using Spin(8) Lie Rotors in Cl(8)."""
    def __init__(self, in_channels: int, num_neurons: int = 1, use_schulz: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_neurons = num_neurons
        self.use_schulz = use_schulz

        self.in_proj = nn.Conv2d(in_channels, num_neurons * 256, kernel_size=1, bias=False)
        self.biv_proj = nn.Conv2d(in_channels, num_neurons * 120, kernel_size=1, bias=False)
        self.metric_proj = nn.Conv2d(in_channels, num_neurons, kernel_size=3, padding=1, bias=False)
        self.gamma_log = nn.Parameter(torch.zeros(num_neurons))
        self.lambda_sw = nn.Parameter(torch.tensor(0.1))
        self.out_proj = nn.Conv2d(num_neurons * 256, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=True)

        triu_idx = torch.triu_indices(16, 16, offset=1)
        self.register_buffer("triu_u", triu_idx[0])
        self.register_buffer("triu_v", triu_idx[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ds = F.softplus(self.metric_proj(x)).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons)
        C_inj = self.in_proj(x).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons, 16, 16)

        biv = self.biv_proj(x).permute(0, 2, 3, 1).contiguous().view(B, H * W, self.num_neurons, 120)
        skew = torch.zeros(B, H * W, self.num_neurons, 16, 16, dtype=biv.dtype, device=x.device)
        skew[..., self.triu_u, self.triu_v] = biv
        skew[..., self.triu_v, self.triu_u] = -biv

        J_sw = 0.5 * (C_inj - C_inj.transpose(-1, -2))
        Omega = (skew + torch.tanh(self.lambda_sw) * J_sw) * (0.5 * ds.unsqueeze(-1).unsqueeze(-1))
        Omega = Omega.clamp(min=-3.0, max=3.0)

        Q = cayley_rotor_16x16(Omega, use_schulz=self.use_schulz)
        gamma = F.softplus(self.gamma_log).view(1, 1, self.num_neurons, 1, 1)
        decay = torch.sigmoid(-gamma * ds.unsqueeze(-1).unsqueeze(-1))
        M = decay * Q

        # Horizontal 2D Scan
        M_h = M.view(B * H, W, self.num_neurons, 16, 16)
        C_h = C_inj.view(B * H, W, self.num_neurons, 16, 16)
        psi_h = (parallel_rotor_scan(M_h, C_h, is_spinor_bundle=True) +
                 parallel_rotor_scan(M_h.flip(1), C_h.flip(1), is_spinor_bundle=True).flip(1)) * 0.5

        # Vertical 2D Scan
        M_v = M.view(B, H, W, self.num_neurons, 16, 16).permute(0, 2, 1, 3, 4, 5).contiguous().view(B * W, H, self.num_neurons, 16, 16)
        C_v = psi_h.view(B, H, W, self.num_neurons, 16, 16).permute(0, 2, 1, 3, 4, 5).contiguous().view(B * W, H, self.num_neurons, 16, 16)
        psi_v = (parallel_rotor_scan(M_v, C_v, is_spinor_bundle=True) +
                 parallel_rotor_scan(M_v.flip(1), C_v.flip(1), is_spinor_bundle=True).flip(1)) * 0.5

        out = psi_v.view(B, W, H, self.num_neurons, 16, 16).permute(0, 2, 1, 3, 4, 5).contiguous()
        out = out.view(B, H, W, self.num_neurons * 256).permute(0, 3, 1, 2).contiguous()
        return self.act(self.norm(x + self.out_proj(out)))


class CliffordFPN(nn.Module):
    """Feature Pyramid Network fusing C3, C4, C5 into multi-scale representations P3, P4, P5."""
    def __init__(self, in_channels_list: List[int], out_channels: int = 128):
        super().__init__()
        c3_c, c4_c, c5_c = in_channels_list
        self.lateral5 = nn.Conv2d(c5_c, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_c, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3_c, out_channels, kernel_size=1)

        self.smooth3 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth4 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth5 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')

        return {
            'p3': self.smooth3(p3),
            'p4': self.smooth4(p4),
            'p5': self.smooth5(p5)
        }


class CliffordDetectionHead(nn.Module):
    """Multi-Scale Decoupled Detection Head (Heatmap, WH, Subpixel Offset)."""
    def __init__(self, in_channels: int = 128, num_classes: int = 1):
        super().__init__()
        self.hm_head = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, p=1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.wh_head = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, p=1),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.off_head = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, p=1),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'heatmap': self.hm_head(x),
            'wh': self.wh_head(x),
            'offset': self.off_head(x)
        }


class CliffordDetModel(nn.Module):
    """Complete Clifford Detection Model (CSN-Large)."""
    def __init__(
        self,
        num_classes: int = 1,
        base_channels: int = 64,
        num_neurons_cl4: int = 2,
        num_neurons_cl8: int = 1,
        use_schulz: bool = True
    ):
        super().__init__()
        self.stem = SpatialPatchStem(in_channels=3, base_channels=base_channels)
        self.clifford_c3 = CliffordSpatialRotorBlockCl4(base_channels * 2, num_neurons=num_neurons_cl4, use_schulz=use_schulz)
        self.clifford_c4 = CliffordSpatialRotorBlockCl8(base_channels * 4, num_neurons=num_neurons_cl8, use_schulz=use_schulz)
        self.clifford_c5 = CliffordSpatialRotorBlockCl8(base_channels * 4, num_neurons=num_neurons_cl8, use_schulz=use_schulz)

        self.fpn = CliffordFPN([base_channels * 2, base_channels * 4, base_channels * 4], out_channels=base_channels * 2)
        self.head_p3 = CliffordDetectionHead(base_channels * 2, num_classes=num_classes)
        self.head_p4 = CliffordDetectionHead(base_channels * 2, num_classes=num_classes)
        self.head_p5 = CliffordDetectionHead(base_channels * 2, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        c3, c4, c5 = self.stem(x)
        c3 = self.clifford_c3(c3)
        c4 = self.clifford_c4(c4)
        c5 = self.clifford_c5(c5)

        fpn_features = self.fpn((c3, c4, c5))
        return {
            'p3': self.head_p3(fpn_features['p3']),
            'p4': self.head_p4(fpn_features['p4']),
            'p5': self.head_p5(fpn_features['p5'])
        }


class ModelEMA:
    """Model Exponential Moving Average with dynamic warmup and BatchNorm buffer synchronization."""
    def __init__(self, model: nn.Module, decay: float = 0.999, updates: int = 0):
        self.ema = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)
            for ema_b, b in zip(self.ema.buffers(), model.buffers()):
                ema_b.copy_(b)
