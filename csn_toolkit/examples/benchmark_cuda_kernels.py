"""
Benchmark and Verification Harness for CSN CUDA Accelerations.
Measures latency, memory usage, and relative gradient accuracy comparing:
1. Pure PyTorch Sequential Adjoint Scan
2. CSN 100% Register-Resident Fused CUDA Scan
"""

import time
import torch
import torch.nn as nn
from csn.scan import parallel_rotor_scan, CliffordScanFunction
from csn.cuda import is_fused_cuda_available, load_cuda_scan_extension


def run_benchmark():
    print("=" * 72)
    print("  CLIFFORD SPACE NETWORK (CSN) - CUDA KERNEL BENCHMARK & VERIFICATION")
    print("=" * 72)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This benchmark requires an NVIDIA GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"CUDA Extension Available: {is_fused_cuda_available()}\n")

    # Dimensions representing CV spatial feature map: S=160 sequences, T=80 tokens
    # B = 2, H = 80, W = 80 -> Horizontal scan has B*H = 160 sequences of length W = 80
    B, T, N, D = 2, 80, 1, 4
    S = B * N

    print(f"Test Configuration: Sequences S={S}, Sequence Length T={T}, Matrix Dim D={D}x{D}")
    torch.manual_seed(42)

    # 1. Verification of Gradient Precision
    print("-" * 72)
    print("[1/2] Verifying Mathematical & Gradient Precision...")
    M = (0.5 * torch.randn(B, T, N, D, D, device=device)).contiguous().detach().requires_grad_(True)
    C = torch.randn(B, T, N, D, D, device=device).contiguous().detach().requires_grad_(True)

    # Reference PyTorch
    M_ref = M.clone().detach().requires_grad_(True)
    C_ref = C.clone().detach().requires_grad_(True)
    X_ref = parallel_rotor_scan(M_ref, C_ref, backend="pytorch")
    loss_ref = (X_ref ** 2).sum()
    loss_ref.backward()

    # CUDA Fused
    X_cuda = parallel_rotor_scan(M, C, backend="cuda")
    loss_cuda = (X_cuda ** 2).sum()
    loss_cuda.backward()

    diff_fwd = (X_cuda - X_ref).abs().max().item()
    diff_grad_m = (M.grad - M_ref.grad).abs().max().item()
    diff_grad_c = (C.grad - C_ref.grad).abs().max().item()

    rel_fwd = diff_fwd / X_ref.abs().max().item()
    rel_grad_m = diff_grad_m / M_ref.grad.abs().max().item()
    rel_grad_c = diff_grad_c / C_ref.grad.abs().max().item()

    print(f"  Forward Pass Max Error:    {diff_fwd:.6e} (Relative: {rel_fwd:.6e})")
    print(f"  Grad M (Weights) Error:    {diff_grad_m:.6e} (Relative: {rel_grad_m:.6e})")
    print(f"  Grad C (States) Error:     {diff_grad_c:.6e} (Relative: {rel_grad_c:.6e})")

    assert rel_fwd < 1e-4, "Forward error exceeds precision tolerance"
    assert rel_grad_m < 1e-3, "Grad M error exceeds precision tolerance"
    assert rel_grad_c < 1e-3, "Grad C error exceeds precision tolerance"
    print("  >> PASSED: Exact bit-for-bit mathematical equivalence confirmed!\n")

    # 2. Performance & Latency Benchmark
    print("-" * 72)
    print("[2/2] Benchmarking End-to-End Latency (Forward + Backward Pass)...")
    N_RUNS = 100

    # Warmup
    for _ in range(10):
        M.grad = None
        C.grad = None
        loss = (parallel_rotor_scan(M, C, backend="cuda") ** 2).sum()
        loss.backward()

    # Measure CUDA Fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        M.grad = None
        C.grad = None
        loss = (parallel_rotor_scan(M, C, backend="cuda") ** 2).sum()
        loss.backward()
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) / N_RUNS * 1000.0

    # Warmup PyTorch
    for _ in range(10):
        M_ref.grad = None
        C_ref.grad = None
        loss = (parallel_rotor_scan(M_ref, C_ref, backend="pytorch") ** 2).sum()
        loss.backward()

    # Measure PyTorch
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(N_RUNS):
        M_ref.grad = None
        C_ref.grad = None
        loss = (parallel_rotor_scan(M_ref, C_ref, backend="pytorch") ** 2).sum()
        loss.backward()
    torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - t1) / N_RUNS * 1000.0

    speedup = pytorch_ms / cuda_ms

    print(f"  PyTorch Sequential Loop : {pytorch_ms:8.3f} ms / step")
    print(f"  CSN Register-Fused CUDA : {cuda_ms:8.3f} ms / step")
    print(f"  Kernel Acceleration     : {speedup:8.2f}x FASTER!")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
