"""
Unit Test: Verify that Parallel Associative Scan matches Sequential Loop exactly.
"""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_scan import parallel_rotor_scan

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_equivalence():
    B = 4
    T = 32
    N = 8
    
    # Generate random Cayley rotors
    # Skew-symmetric matrices
    skew = torch.randn(B, T, N, 4, 4, device=device)
    skew = 0.5 * (skew - skew.transpose(-1, -2))
    
    eye = torch.eye(4, device=device).view(1, 1, 1, 4, 4)
    r_num = eye + skew
    r_rev = eye - skew
    denom = 1.0 + 0.25 * (skew ** 2).sum(dim=(-1, -2), keepdim=True)
    Q = torch.matmul(r_num, r_rev) / denom # [B, T, N, 4, 4]
    
    decay = torch.sigmoid(torch.randn(B, T, N, 1, 1, device=device))
    M = torch.sqrt(decay) * Q # [B, T, N, 4, 4]
    
    S = torch.randn(B, T, N, 4, 4, device=device)
    C = (1.0 - decay) * S # [B, T, N, 4, 4]
    
    # 1. Sequential computation:
    seq_X = []
    x_prev = torch.zeros(B, N, 4, 4, device=device)
    for t in range(T):
        m_t = M[:, t]
        c_t = C[:, t]
        # x_next = m_t @ x_prev @ m_t^T + c_t
        x_next = torch.matmul(torch.matmul(m_t, x_prev), m_t.transpose(-1, -2)) + c_t
        seq_X.append(x_next)
        x_prev = x_next
    seq_X = torch.stack(seq_X, dim=1)
    
    # 2. Parallel Associative Scan:
    par_X = parallel_rotor_scan(M, C)
    
    # 3. Check difference:
    max_diff = torch.max(torch.abs(seq_X - par_X)).item()
    print(f"Maximum difference between Sequential and Parallel Scan: {max_diff:e}")
    assert max_diff < 1e-4, f"Failed! Difference too large: {max_diff}"
    print("[PASS] Parallel Associative Scan matches sequential loop exactly!")

if __name__ == '__main__':
    test_equivalence()
