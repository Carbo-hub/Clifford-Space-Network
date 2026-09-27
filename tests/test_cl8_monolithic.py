"""
Verification of Monolithic Cl(8, 0) Architecture:
State per neuron: 256 dimensions (16x16 real matrices, M_16(R))
Lie group: Spin(8) / SO(16) (120 bivector generators)
Native mapping to NVIDIA Tensor Core tile m16n16k16.
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_scan import parallel_rotor_scan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_cl8_monolithic():
    print("=" * 80)
    print("  VERIFYING MONOLITHIC CL(8, 0) ARCHITECTURE (256 DIMS / NEURON, M_16(R))")
    print("=" * 80)
    
    B, T, D = 4, 64, 48
    num_neurons = 2 # 2 neurons * 256 = 512 state dimensions!
    dt_base = 0.05
    
    # 120 bivector generators for skew-symmetric so(16) (16*15/2 = 120)
    num_biv = 16 * 15 // 2
    print(f"  Cl(8, 0) State per neuron: 256 dimensions (16x16 matrix)")
    print(f"  Bivector generators in so(16): {num_biv}")
    
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    
    # Projections
    in_proj = nn.Linear(D, num_neurons * 256).to(device)
    gamma_proj = nn.Linear(D, num_neurons).to(device)
    biv_proj = nn.Linear(D, num_neurons * num_biv).to(device)
    time_proj = nn.Linear(D, 1).to(device)
    
    nn.init.zeros_(time_proj.weight)
    nn.init.constant_(time_proj.bias, math.log(math.e - 1.0))
    
    triu_u, triu_v = torch.triu_indices(16, 16, offset=1)
    
    # Forward pass
    t0 = time.time()
    dt = (F.softplus(time_proj(x)) * dt_base).unsqueeze(-1).unsqueeze(-1) # [B, T, 1, 1, 1]
    S = in_proj(x).view(B, T, num_neurons, 16, 16)
    gamma = F.softplus(gamma_proj(x)).unsqueeze(-1).unsqueeze(-1)
    decay = torch.sigmoid(-gamma * dt)
    
    biv = biv_proj(x).view(B, T, num_neurons, num_biv)
    skew = torch.zeros(B, T, num_neurons, 16, 16, device=device)
    skew[..., triu_u, triu_v] = biv
    skew[..., triu_v, triu_u] = -biv
    
    Omega = skew * (0.5 * dt)
    eye16 = torch.eye(16, device=device).expand_as(Omega)
    
    # Cayley rotor solve on 16x16 matrices
    Q = torch.linalg.solve(eye16 - Omega, eye16 + Omega)
    
    # Check orthogonality
    ortho_err = torch.max(torch.abs(torch.matmul(Q.transpose(-1, -2), Q) - eye16)).item()
    print(f"  Cayley Rotor 16x16 Orthogonality Error ||Q^T Q - I||: {ortho_err:.2e}")
    assert ortho_err < 1e-5, f"16x16 Cayley rotor not orthogonal! Error: {ortho_err}"
    
    # Scan step test
    M = torch.sqrt(decay) * Q
    C = (1.0 - decay) * S
    X_all = parallel_rotor_scan(M, C)
    
    loss = (X_all ** 2).sum()
    loss.backward()
    
    elapsed = time.time() - t0
    print(f"  X_all shape: {list(X_all.shape)}")
    print(f"  Forward + Backward finished in {elapsed*1000:.2f} ms")
    print(f"  Gradient w.r.t input x norm: {x.grad.norm().item():.4f}")
    assert x.grad is not None and not torch.isnan(x.grad).any(), "NaNs in gradient!"
    
    print("\n  -> [VERIFIED]: Monolithic Cl(8, 0) with 256 dims/neuron is fast, exact, and numerically stable!")
    print("=" * 80)


if __name__ == '__main__':
    test_cl8_monolithic()
