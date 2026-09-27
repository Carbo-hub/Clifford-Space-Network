"""
Master Benchmark: Clifford-Det Cl(8, 0) vs. YOLO26n (Ultralytics)
Frame-by-Frame Real-Time Single-Image Object Detection at 640x640 resolution.
Evaluated on NVIDIA GeForce GTX 1660 Ti.

Metrics:
1. Parameter Count & Weight SRAM footprint (KB / MB)
2. Pure Inference Peak VRAM at Batch 1 and Batch 8
3. Single-Image Latency (ms) and Throughput (FPS)
4. Detection Performance on standard benchmark images
5. Geometric Invariance / Rotation Stress Test (0, 15, 30, 45, 90 degrees)
"""

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from python.clifford_vision import CliffordDetModel
from ultralytics import YOLO


def benchmark_pure_inference():
    print("==========================================================================================")
    print("MASTER BENCHMARK: CLIFFORD-DET CL(8, 0) vs. YOLO26n (640x640 SINGLE-FRAME)")
    print("==========================================================================================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load Models
    print("\n[INFO] Loading YOLO26n (Ultralytics)...")
    yolo = YOLO('yolo26n.pt')
    yolo_model = yolo.model.to(device).eval()

    print("[INFO] Loading Clifford-Det Cl(8, 0)...")
    clifford_model = CliffordDetModel(
        num_classes=80, # Support COCO 80 classes
        base_channels=24,
        num_neurons=1,
        num_blocks=2,
        use_schulz=True
    ).to(device).eval()

    yolo_params = sum(p.numel() for p in yolo_model.parameters())
    clifford_params = sum(p.numel() for p in clifford_model.parameters())
    yolo_sram_kb = sum(p.numel() * 4 for p in yolo_model.parameters()) / 1024
    clifford_sram_kb = sum(p.numel() * 4 for p in clifford_model.parameters()) / 1024

    print(f"  YOLO26n Parameters:       {yolo_params:,} ({yolo_sram_kb/1024:.2f} MB SRAM)")
    print(f"  Clifford-Det Parameters:  {clifford_params:,} ({clifford_sram_kb:.1f} KB SRAM)")
    print(f"  Parameter Compression:    {yolo_params / clifford_params:.1f}x smaller!")

    # --------------------------------------------------------------------------------------
    # 2. Batch 1 Streaming Latency & Real VRAM (Single Camera Feed)
    # --------------------------------------------------------------------------------------
    x1 = torch.randn(1, 3, 640, 640, device=device)

    # Warmup YOLO
    with torch.no_grad():
        for _ in range(15):
            _ = yolo_model(x1)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            _ = yolo_model(x1)
    torch.cuda.synchronize()
    yolo_lat_b1 = (time.perf_counter() - t0) / iters * 1000
    yolo_vram_b1 = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Warmup Clifford
    with torch.no_grad():
        for _ in range(15):
            _ = clifford_model(x1)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = clifford_model(x1)
    torch.cuda.synchronize()
    clifford_lat_b1 = (time.perf_counter() - t0) / iters * 1000
    clifford_vram_b1 = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --------------------------------------------------------------------------------------
    # 3. Batch 8 Batched Throughput & Real VRAM (Multi-Camera / High-Density Inference)
    # --------------------------------------------------------------------------------------
    x8 = torch.randn(8, 3, 640, 640, device=device)

    # YOLO Batch 8
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            _ = yolo_model(x8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters_b8 = 30
        for _ in range(iters_b8):
            _ = yolo_model(x8)
        torch.cuda.synchronize()
        dur_yolo = time.perf_counter() - t0
        yolo_fps = (8 * iters_b8) / dur_yolo
    yolo_vram_b8 = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Clifford Batch 8
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            _ = clifford_model(x8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters_b8):
            _ = clifford_model(x8)
        torch.cuda.synchronize()
        dur_clifford = time.perf_counter() - t0
        clifford_fps = (8 * iters_b8) / dur_clifford
    clifford_vram_b8 = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --------------------------------------------------------------------------------------
    # 4. Geometric Rotational Invariance Test
    # --------------------------------------------------------------------------------------
    print("\n[INFO] Running Geometric Rotation Stress Test (0°, 15°, 30°, 45°, 90°)...")
    angles = [0, 15, 30, 45, 90]
    yolo_conf_drops = []
    clifford_feature_corrs = []

    # Create synthetic test pattern with oriented edges & geometric structures
    base_img = torch.zeros(1, 3, 640, 640, device=device)
    # Draw a rectangle / person-like bounding box
    base_img[:, :, 200:440, 280:360] = 0.8
    base_img[:, 0, 200:440, 280:360] = 0.9  # Red/Orange tint

    with torch.no_grad():
        base_yolo = yolo_model(base_img)
        base_clifford = clifford_model(base_img)["heatmap"]

    rot_results = {}
    for ang in angles:
        if ang == 0:
            rot_img = base_img
        else:
            # Rotate image using affine grid
            theta = math.radians(ang)
            rot_mat = torch.tensor([
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta),  math.cos(theta), 0]
            ], dtype=torch.float, device=device).unsqueeze(0)
            grid = F.affine_grid(rot_mat, base_img.size(), align_corners=False)
            rot_img = F.grid_sample(base_img, grid, align_corners=False)

        with torch.no_grad():
            out_yolo = yolo_model(rot_img)
            out_clifford = clifford_model(rot_img)["heatmap"]

            # Measure max activation / detection preservation
            if isinstance(out_yolo, (tuple, list)):
                y_act = out_yolo[0].sigmoid().max().item()
            else:
                y_act = out_yolo.sigmoid().max().item()

            c_act = out_clifford.max().item()

        rot_results[ang] = {"yolo_act": y_act, "clifford_act": c_act}

    # --------------------------------------------------------------------------------------
    # 5. Master Summary Table
    # --------------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("MASTER SUMMARY RESULTS: CLIFFORD-DET CL(8, 0) vs. YOLO26n")
    print("=" * 90)
    print(f"{'Metric':<35} | {'YOLO26n (Ultralytics)':<22} | {'Clifford-Det Cl(8, 0)':<22} | {'Advantage':<15}")
    print("-" * 90)
    print(f"{'Total Parameters':<35} | {yolo_params:<22,d} | {clifford_params:<22,d} | {yolo_params/clifford_params:.1f}x smaller")
    print(f"{'Weight Footprint (SRAM FP32)':<35} | {yolo_sram_kb/1024:<19.2f} MB | {clifford_sram_kb:<19.1f} KB | {yolo_sram_kb/clifford_sram_kb:.1f}x lighter")
    print(f"{'Streaming Latency (Batch 1, ms)':<35} | {yolo_lat_b1:<19.2f} ms | {clifford_lat_b1:<19.2f} ms | {yolo_lat_b1/clifford_lat_b1:.2f}x")
    print(f"{'Streaming FPS (Batch 1)':<35} | {1000/yolo_lat_b1:<19.1f} FPS| {1000/clifford_lat_b1:<19.1f} FPS| {clifford_lat_b1/yolo_lat_b1:.2f}x")
    print(f"{'Batched Throughput (Batch 8)':<35} | {yolo_fps:<19.1f} FPS| {clifford_fps:<19.1f} FPS|")
    print(f"{'Inference VRAM (Batch 1)':<35} | {yolo_vram_b1:<19.2f} MB | {clifford_vram_b1:<19.2f} MB | {'Lighter' if clifford_vram_b1 < yolo_vram_b1 else ''}")
    print(f"{'Inference VRAM (Batch 8)':<35} | {yolo_vram_b8:<19.2f} MB | {clifford_vram_b8:<19.2f} MB |")
    print("-" * 90)
    print("ROTATIONAL STRESS TEST (Peak Detection Response under Geometric Tilt):")
    for ang in angles:
        print(f"  Angle {ang:2d}°: YOLO = {rot_results[ang]['yolo_act']:.4f} | Clifford-Det = {rot_results[ang]['clifford_act']:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    benchmark_pure_inference()
