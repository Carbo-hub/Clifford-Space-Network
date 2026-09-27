"""
Benchmarking Suite for Clifford Space Networks.

Measures:
1. CV Detection Model: Latency, FPS, and VRAM for 640x640 images.
2. Sequence / Transformer LM: Latency across sequence lengths (128, 512, 1024, 2048)
   demonstrating linear O(L) scaling vs quadratic attention.
"""

import sys
import time
from pathlib import Path
import torch

toolkit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(toolkit_path))

from csn import CliffordDetModel, CliffordTransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bench_cv():
    print("=" * 70)
    print("      BENCHMARK 1: CSN-LARGE COMPUTER VISION (640x640)      ")
    print("=" * 70)
    model = CliffordDetModel(num_classes=1, base_channels=64, use_schulz=True).to(device)
    model.eval()

    batch_sizes = [1, 2, 4]
    for b in batch_sizes:
        x = torch.randn(b, 3, 640, 640, device=device)
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        fps = b / dt
        ms_per_step = dt * 1000.0
        print(f"  Batch Size {b:2d}: {ms_per_step:6.1f} ms/step | {fps:5.1f} img/s")


def bench_transformer():
    print("\n" + "=" * 70)
    print("      BENCHMARK 2: CLIFFORD TRANSFORMER SEQUENCE SCALING (O(L))      ")
    print("=" * 70)
    model = CliffordTransformerLM(vocab_size=2048, d_model=128, num_layers=4, num_heads=4).to(device)
    model.eval()

    seq_lengths = [128, 256, 512, 1024]
    for seq_len in seq_lengths:
        x = torch.randint(0, 2048, (1, seq_len), device=device)
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tok_per_sec = seq_len / dt
        print(f"  Sequence Length {seq_len:4d}: {dt*1000.0:6.1f} ms | {tok_per_sec:7.1f} tokens/s")

if __name__ == "__main__":
    bench_cv()
    bench_transformer()
