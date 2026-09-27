"""
Unit Test Suite & Verification:
Closed-Form Schulz Inversion & Hardware-Optimized Spinor Bundle Cl(8, 0)
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath('.'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from python.clifford_cl8 import cayley_rotor_16x16, CliffordCL8Block, CliffordCL8ForCausalLM
from python.clifford_scan import parallel_rotor_scan


class TestOptimizedCliffordCL8(unittest.TestCase):

    def test_schulz_rotor_orthogonality_and_equivalence(self):
        """Test that Schulz GEMM inversion matches torch.linalg.solve and produces orthogonal rotors."""
        B, T, N = 8, 32, 1
        u, v = torch.triu_indices(16, 16, offset=1)
        biv = torch.randn(B, T, N, 120, device=device) * 0.05
        skew = torch.zeros(B, T, N, 16, 16, device=device)
        skew[..., u, v] = biv
        skew[..., v, u] = -biv
        Omega = skew * 0.05

        eye = torch.eye(16, device=device).expand_as(Omega)

        # 1. Classical LU solve
        Q_solve = cayley_rotor_16x16(Omega, use_schulz=False)
        # 2. Schulz iterative GEMM inversion
        Q_schulz = cayley_rotor_16x16(Omega, use_schulz=True)

        # Numerical equivalence
        diff = torch.norm(Q_solve - Q_schulz) / torch.norm(Q_solve)
        self.assertLess(diff.item(), 1e-6, f"Schulz vs Solve mismatch: {diff.item()}")

        # Exact orthogonality Q^T Q = I
        ortho_err = torch.norm(torch.matmul(Q_schulz.transpose(-1, -2), Q_schulz) - eye) / eye.numel()
        self.assertLess(ortho_err.item(), 1e-6, f"Schulz non-orthogonal: {ortho_err.item()}")
        print(f"\n[PASS] Schulz Rotor Equivalence (diff={diff.item():.2e}) & Orthogonality (err={ortho_err.item():.2e})")

    def test_spinor_bundle_scan_associativity(self):
        """Test that one-sided spinor bundle scan matches sequential recurrence Psi_{t+1} = M_t Psi_t + C_t."""
        B, T, N = 4, 32, 1
        M = torch.randn(B, T, N, 16, 16, device=device) * 0.1
        C = torch.randn(B, T, N, 16, 16, device=device)

        # Sequential reference
        psi = torch.zeros(B, N, 16, 16, device=device)
        seq_outs = []
        for t in range(T):
            psi = torch.matmul(M[:, t], psi) + C[:, t]
            seq_outs.append(psi)
        seq_traj = torch.stack(seq_outs, dim=1)

        # Parallel Scan (is_spinor_bundle=True)
        scan_traj = parallel_rotor_scan(M, C, is_spinor_bundle=True)

        diff = torch.norm(seq_traj - scan_traj) / torch.norm(seq_traj)
        self.assertLess(diff.item(), 1e-5, f"Spinor bundle scan mismatch with sequential: {diff.item()}")
        print(f"[PASS] Spinor Bundle Scan Matches Sequential Recurrence (diff={diff.item():.2e})")

    def test_autograd_backward_gradient_flow(self):
        """Test that gradients flow smoothly to all parameters without NaN or Inf."""
        model = CliffordCL8ForCausalLM(
            vocab_size=65,
            d_model=32,
            num_layers=2,
            num_neurons=1,
            use_schulz=True,
            is_spinor_bundle=True
        ).to(device)

        x = torch.randint(0, 65, (4, 32), device=device)
        y = torch.randint(0, 65, (4, 32), device=device)

        logits, loss, diags = model(x, targets=y)
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in {name}")

        print(f"[PASS] Gradient Flow Verified: All parameters received clean, finite gradients!")

    def test_speedup_benchmark(self):
        """Compare forward pass speed between baseline and optimized Schulz + Spinor Bundle."""
        import time
        model_baseline = CliffordCL8ForCausalLM(
            vocab_size=65, d_model=48, num_layers=2, num_neurons=1,
            use_schulz=False, is_spinor_bundle=False
        ).to(device)
        model_baseline.eval()

        model_optimized = CliffordCL8ForCausalLM(
            vocab_size=65, d_model=48, num_layers=2, num_neurons=1,
            use_schulz=True, is_spinor_bundle=True
        ).to(device)
        model_optimized.eval()

        x = torch.randint(0, 65, (32, 128), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_baseline(x)
                _ = model_optimized(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        iters = 50
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model_baseline(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_base = (time.perf_counter() - t0) * 1000.0 / iters

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model_optimized(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_opt = (time.perf_counter() - t0) * 1000.0 / iters

        speedup = t_base / t_opt
        print(f"[BENCHMARK] Baseline: {t_base:.2f} ms | Optimized: {t_opt:.2f} ms | Speedup: {speedup:.2f}x ({((speedup - 1) * 100):.1f}% faster!)")
        self.assertGreater(speedup, 1.05)


if __name__ == '__main__':
    unittest.main()
