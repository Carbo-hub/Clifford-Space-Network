"""
Clifford-LTC / TENN PyTorch Module and CUDA Extension Interface

This module provides a drop-in PyTorch recurrent layer (CliffordTENNCell)
implementing continuous-time liquid dynamics over Clifford algebra Cl(4, 0).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class CliffordTENNCell(nn.Module):
    """
    Clifford-LTC / TENN Microcircuit Cell.
    
    Args:
        num_neurons (int): Number of multivector neurons (e.g. 16, 32, 64).
        in_dim (int): Dimensionality of external input.
        algebra_dim (int): Multivector dimension (default: 16 for Cl(4, 0)).
    """
    def __init__(self, num_neurons: int = 32, in_dim: int = 8, algebra_dim: int = 16):
        super().__init__()
        self.num_neurons = num_neurons
        self.in_dim = in_dim
        self.algebra_dim = algebra_dim  # 16 components for Cl(4,0)

        # Recurrent synaptic multivector weights: [N, N, 16]
        # Initialized with small random weights to ensure contractivity
        self.W_rec = nn.Parameter(
            torch.randn(num_neurons, num_neurons, algebra_dim) * (0.05 / math.sqrt(num_neurons))
        )

        # Input projection multivector weights: [N, in_dim, 16]
        self.W_in = nn.Parameter(
            torch.randn(num_neurons, in_dim, algebra_dim) * (0.1 / math.sqrt(in_dim))
        )

        # Baseline leak conductance (softplus argument)
        self.gamma_bias = nn.Parameter(torch.zeros(num_neurons))

        # Baseline bivector torques (6 components for Cl(4,0) bivectors)
        self.bivector_bias = nn.Parameter(torch.zeros(num_neurons, 6))

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns zero-initialized multivector state [batch_size, num_neurons, 16]"""
        return torch.zeros(batch_size, self.num_neurons, self.algebra_dim, device=device)

    def forward(
        self, 
        x_t: torch.Tensor, 
        prev_state: torch.Tensor, 
        dt: float = 0.02
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward step of the Clifford-LTC microcircuit.
        
        Args:
            x_t: Input tensor [batch_size, in_dim]
            prev_state: State tensor [batch_size, num_neurons, 16]
            dt: Continuous time elapsed since last step (in seconds)
            
        Returns:
            output: Readout vector or full multivector state [batch_size, num_neurons, 16]
            next_state: Updated state [batch_size, num_neurons, 16]
        """
        # When compiled CUDA kernel extension is loaded, it dispatches directly:
        # return clifford_cuda.forward(prev_state, x_t, self.W_rec, self.W_in, self.gamma_bias, self.bivector_bias, dt)
        
        # PyTorch reference implementation of the CUDA kernel logic:
        B, N, D = prev_state.shape
        
        # 1. Synaptic interference: S_i = sum_j (W_rec[i, j] * X_j) + W_in * I
        # For Cl(4,0), represented via 4x4 matrix multiplication
        prev_mat = prev_state.view(B, N, 4, 4)
        w_mat = self.W_rec.view(N, N, 4, 4)
        
        # Einstein summation: B = batch, i = post, j = pre, r = row, c = col, k = contract
        synaptic = torch.einsum('ijrk,bjkc->birc', w_mat, prev_mat) # [B, N, 4, 4]
        
        # Input projection
        w_in_mat = self.W_in.view(N, self.in_dim, 4, 4)
        in_proj = torch.einsum('nikr,bi->bnkr', w_in_mat, x_t) # [B, N, 4, 4]
        
        S = synaptic + in_proj
        
        # 2. Extract liquid conductance gamma (trace / scalar part)
        trace_S = S[:, :, 0, 0] + S[:, :, 1, 1] + S[:, :, 2, 2] + S[:, :, 3, 3]
        gamma = torch.nn.functional.softplus(self.gamma_bias + 0.25 * trace_S)
        
        # 3. Cayley transform rotation of previous state:
        # Extract skew-symmetric bivector part: Omega = 0.5 * (S - S^T)
        Omega = 0.5 * (S - S.transpose(-1, -2)) * (0.5 * dt)
        
        eye = torch.eye(4, device=prev_state.device).unsqueeze(0).unsqueeze(0)
        r_num = eye + Omega
        r_rev = eye - Omega
        
        omega_norm_sq = 0.5 * torch.sum(Omega ** 2, dim=(-1, -2), keepdim=True)
        denom = 1.0 + 0.25 * omega_norm_sq
        
        # Fast isometric Cayley rotation: Q = (I + Omega)(I - Omega) / denom
        Q = torch.matmul(r_num, r_rev) / denom
        
        # State rotation: X_rot = Q * X_prev * Q^T
        X_rot = torch.matmul(torch.matmul(Q, prev_mat), Q.transpose(-1, -2))
        
        # 4. Liquid Time-Constant gating (decay):
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1)
        
        next_mat = decay * X_rot + (1.0 - decay) * S
        next_state = next_mat.view(B, N, D)
        
        return next_state, next_state
