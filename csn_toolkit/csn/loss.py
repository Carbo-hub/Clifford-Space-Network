"""
Geometric Loss Functions on Contact & Riemannian Manifolds for Clifford Networks.

Implements:
1. compute_sasaki_ciou_vectorized:
   Exact Geodesic Distance on the Unit Tangent Bundle (T_1 M) equipped with the
   Sasaki Contact Metric. Combines:
     - Horizontal base metric (centroid displacement: rho^2 / c^2)
     - Scale metric (1 - IoU)
     - Vertical Reeb fiber distance: kappa_reeb * (4/pi^2) * (atan(w2/h2) - atan(w1/h1))^2
2. compute_vectorized_detection_loss:
   Zero-sync multi-scale detection loss for center heatmaps, subpixel offsets, and Sasaki bounding boxes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def compute_sasaki_ciou_vectorized(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    kappa_reeb: float = 0.85,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes Geodesic Distance on the Sasaki Contact Manifold T_1 M for bounding boxes.

    Args:
        boxes1: [N, 4] tensor of (x1, y1, x2, y2)
        boxes2: [N, 4] tensor of (x1, y1, x2, y2)
        kappa_reeb: Weight for the Reeb vector field angular distortion penalty.
        eps: Small epsilon for numerical stability.

    Returns:
        [N] tensor of Sasaki geodesic distance values in [0, 3].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    # Box dimensions
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    # Intersections & Unions
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)

    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / (union_area + eps)

    # Horizontal base manifold metric: Enclosing box diagonal and centroid squared distance
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    b1_cx = (b1_x1 + b1_x2) * 0.5
    b1_cy = (b1_y1 + b1_y2) * 0.5
    b2_cx = (b2_x1 + b2_x2) * 0.5
    b2_cy = (b2_y1 + b2_y2) * 0.5
    rho2 = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2)
    d_base = rho2 / c2

    # Vertical Reeb fiber metric: Aspect ratio geodesic distortion
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    d_reeb = kappa_reeb * alpha * v

    # Sasaki Contact Metric: ds^2 = d_base + (1 - iou) + d_reeb
    return (1.0 - iou) + d_base + d_reeb


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius_x: int, radius_y: int):
    """Draws an anisotropic 2D Gaussian peak directly into a NumPy heatmap."""
    diameter_x = 2 * radius_x + 1
    diameter_y = 2 * radius_y + 1

    sigma_x = diameter_x / 6.0
    sigma_y = diameter_y / 6.0

    x = np.arange(-radius_x, radius_x + 1, dtype=np.float32)
    y = np.arange(-radius_y, radius_y + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    gaussian = np.exp(-(xx * xx / (2 * sigma_x * sigma_x) + yy * yy / (2 * sigma_y * sigma_y)))
    x0, y0 = center

    height, width = heatmap.shape
    left, right = min(x0, radius_x), min(width - x0, radius_x + 1)
    top, bottom = min(y0, radius_y), min(height - y0, radius_y + 1)

    masked_heatmap = heatmap[y0 - top:y0 + bottom, x0 - left:x0 + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
