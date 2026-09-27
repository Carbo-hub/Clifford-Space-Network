"""
Clifford-TENN Cl(6, 0): Monolithic 64-Dimensional Clifford Continuous-Time Architecture
with Adaptive Continuous-Time Warping (Delta t(t)) and Exact Cayley Rotors on M_8(R).

Mathematical Core:
1. Multivector Space: Cl(6, 0) isomorphic to real 8x8 matrices M_8(R) (64 state dimensions).
2. Adaptive Time Pacing: Delta t_t = softplus(W_dt * x_t + b_dt) * dt_base.
3. Lie Algebra so(8) / so(6): 28 bivector generators producing skew-symmetric Omega_t.
4. Exact Cayley Rotor: Q_t = (I - Omega_t)^(-1) (I + Omega_t) in Spin(6) = SU(4) / SO(8).
   Orthogonality Q^T Q = I holds unconditionally without singularities.
5. Associative Parallel Rotor Scan in O(log T) time (verified in Lean 4).
6. Neuromorphic Event Spikes (Surrogate Heaviside on Frobenius norm ||X||_F).
7. Geometric Gated Linear Unit (GGLU).
8. Canonical Riemannian Geodesic Head with Bayesian Token Prior Calibration.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.clifford_scan import parallel_rotor_scan, surrogate_spike
from python.clifford_model import RiemannianGeodesicHead, GeometricGatedLinearUnit


class AdaptiveTimeHead(nn.Module):
    """
    Adaptive Continuous-Time Pacing Head (Continuous Dynamic Time Warping).
    Predicts token-dependent time step Delta t_t:
        Delta t_t = softplus(W_dt * x_t + b_dt) * dt_base
    Initialized so that softplus(b_dt) = 1.0 (starts exactly at dt_base).
    """
    def __init__(self, d_model: int, dt_base: float = 0.05):
        super().__init__()
        self.dt_base = dt_base
        self.proj = nn.Linear(d_model, 1)
        
        # Initialize W_dt near zero and b_dt such that softplus(b_dt) = 1.0
        # softplus(x) = ln(1 + e^x) = 1.0 => x = ln(e - 1) approx 0.54132485
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, math.log(math.e - 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            dt: [B, T, 1, 1, 1] strictly positive time steps
        """
        # softplus guarantees Delta t > 0 strictly
        scale = F.softplus(self.proj(x)) # [B, T, 1]
        dt = scale * self.dt_base
        return dt.unsqueeze(-1).unsqueeze(-1) # [B, T, 1, 1, 1]


def cayley_rotor_8x8(Omega: torch.Tensor) -> torch.Tensor:
    """
    Computes exact Cayley transform for 8x8 skew-symmetric matrices:
        Q = (I - Omega)^(-1) (I + Omega)
    strictly in Spin(6) / SO(8) with Q^T Q = I.
    Args:
        Omega: [..., 8, 8] skew-symmetric tensor
    """
    eye = torch.eye(8, device=Omega.device).expand_as(Omega)
    # torch.linalg.solve solves (I - Omega) Q = (I + Omega) with zero singular points
    Q = torch.linalg.solve(eye - Omega, eye + Omega)
    return Q


class CliffordCL6Block(nn.Module):
    """
    Hierarchical Monolithic Cl(6, 0) Layer with Adaptive Time Pacing.
    
    State per neuron: 64 continuous multivector dimensions (8x8 real matrix).
    """
    def __init__(
        self,
        d_model: int,
        num_neurons: int = 6,
        kernel_size: int = 4,
        dt_base: float = 0.05,
        spike_threshold: float = 0.1,
        spike_beta: float = 10.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.dt_base = dt_base
        self.spike_threshold = spike_threshold
        self.spike_beta = spike_beta
        
        # 1. Pre-Normalization
        self.ln_1 = nn.LayerNorm(d_model)
        
        # 2. Causal 1D Temporal Front
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        
        # 3. Adaptive Time Pacing Head
        self.time_head = AdaptiveTimeHead(d_model, dt_base=dt_base)
        
        # 4. Clifford Multivector Projections in Cl(6, 0)
        # 64 components per neuron (8x8 matrix)
        self.in_proj = nn.Linear(d_model, num_neurons * 64)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        # 28 bivector generators for skew-symmetric so(8) / Spin(6)
        self.biv_proj = nn.Linear(d_model, num_neurons * 28)
        self.omega_scale = nn.Parameter(torch.ones(num_neurons, 1) * 10.0)
        
        # 5. Inter-neuron Clifford Mixing
        self.W_mix = nn.Parameter(torch.randn(num_neurons, num_neurons) / math.sqrt(num_neurons))
        self.out_proj = nn.Linear(num_neurons * 64, d_model)
        
        # 6. GGLU Feedforward
        self.ln_2 = nn.LayerNorm(d_model)
        self.gglu = GeometricGatedLinearUnit(d_model, d_ffn=2 * d_model)
        
        # Precomputed upper-triangle indices for 8x8 skew-symmetric matrix (28 elements)
        self.register_buffer('triu_u', torch.triu_indices(8, 8, offset=1)[0], persistent=False)
        self.register_buffer('triu_v', torch.triu_indices(8, 8, offset=1)[1], persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            out: [B, T, d_model]
            Omega: [B, T, num_neurons, 8, 8] skew-symmetric generator tensor
            dt: [B, T, 1] learned adaptive time steps
        """
        B, T, D = x.shape
        residual = x
        
        x_norm = self.ln_1(x)
        x_conv = self.conv1d(x_norm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 1. Predict Adaptive Time Step Delta t_t for each token
        dt = self.time_head(x_conv) # [B, T, 1, 1, 1]
        
        # 2. Injected multivector field S in M_8(R)
        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 8, 8)
        
        # 3. Continuous-time decay with adaptive time: decay = sigmoid(-gamma * dt_t)
        gamma = F.softplus(self.gamma_proj(x_conv)).unsqueeze(-1).unsqueeze(-1) # [B, T, N, 1, 1]
        decay = torch.sigmoid(-gamma * dt) # [B, T, N, 1, 1]
        
        # 4. Construct 8x8 Skew-Symmetric Generator Omega_t
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 28) * self.omega_scale
        skew = torch.zeros(B, T, self.num_neurons, 8, 8, device=x.device)
        skew[..., self.triu_u, self.triu_v] = biv
        skew[..., self.triu_v, self.triu_u] = -biv
        
        # Scale by adaptive time step: Omega_t = skew * (0.5 * dt_t)
        Omega = skew * (0.5 * dt)
        
        # 5. Exact Cayley Rotor Q_t in Spin(6) / SO(8)
        Q = cayley_rotor_8x8(Omega)
        
        # 6. Parallel Associative Scan Operators
        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S
        
        # O(log T) Parallel Prefix-Scan
        X_all = parallel_rotor_scan(M, C)
        
        # 7. Neuromorphic Event Spiking
        energy = torch.norm(X_all, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=self.spike_threshold, beta=self.spike_beta)
        X_spiked = X_all * spikes
        
        # 8. Inter-neuron Clifford Mixing
        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)
        feat = X_mixed.reshape(B, T, self.num_neurons * 64)
        
        # 9. Residual stream + GGLU
        h = residual + self.out_proj(feat)
        out = h + self.gglu(self.ln_2(h))
        
        return out, Omega, dt.squeeze(-1).squeeze(-1).squeeze(-1) # return dt as [B, T]


class CliffordCL6ForCausalLM(nn.Module):
    """
    Complete Deep Causal Language Model with Monolithic Cl(6, 0) Blocks,
    Adaptive Continuous-Time Pacing, and Canonical Riemannian Geodesic Head.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        num_layers: int = 2,
        num_neurons: int = 6,
        dt_base: float = 0.05,
        kernel_size: int = 4,
        spike_threshold: float = 0.1,
        lambda_kin: float = 0.01,
        lambda_smooth: float = 0.005,
        lambda_time: float = 0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.dt_base = dt_base
        self.lambda_kin = lambda_kin
        self.lambda_smooth = lambda_smooth
        self.lambda_time = lambda_time
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            CliffordCL6Block(
                d_model=d_model,
                num_neurons=num_neurons,
                kernel_size=kernel_size,
                dt_base=dt_base,
                spike_threshold=spike_threshold
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = RiemannianGeodesicHead(vocab_size=vocab_size, d_model=d_model)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """
        Args:
            idx: [B, T] token indices
            targets: [B, T] target token indices (optional)
        Returns:
            logits: [B, T, vocab_size]
            loss: total scalar loss (CE + Action + Time Smoothness)
            diagnostics: dictionary with action_loss, time_loss, mean_dt, min_dt, max_dt
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        total_e_kin = 0.0
        total_e_smooth = 0.0
        total_time_loss = 0.0
        all_dts = []
        
        for block in self.blocks:
            x, Omega, dt_t = block(x) # dt_t: [B, T]
            all_dts.append(dt_t)
            
            # Hamiltonian Least-Action Regularization
            e_kin = (Omega ** 2).mean()
            if T > 1:
                e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
                # Clock smoothness penalty: penalizes high-frequency temporal acceleration
                time_smooth = (((dt_t[:, 1:] - dt_t[:, :-1]) / self.dt_base) ** 2).mean()
            else:
                e_smooth = torch.tensor(0.0, device=idx.device)
                time_smooth = torch.tensor(0.0, device=idx.device)
                
            total_e_kin = total_e_kin + e_kin
            total_e_smooth = total_e_smooth + e_smooth
            total_time_loss = total_time_loss + time_smooth
            
        x_norm = self.ln_f(x)
        logits = self.head(x_norm)
        
        loss = None
        diagnostics = {}
        
        stacked_dt = torch.stack(all_dts)
        mean_dt = stacked_dt.mean().item()
        min_dt = stacked_dt.min().item()
        max_dt = stacked_dt.max().item()
        
        diagnostics['mean_dt'] = mean_dt
        diagnostics['min_dt'] = min_dt
        diagnostics['max_dt'] = max_dt
        
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            action_penalty = self.lambda_kin * total_e_kin + self.lambda_smooth * total_e_smooth
            time_penalty = self.lambda_time * total_time_loss
            loss = ce_loss + action_penalty + time_penalty
            
            diagnostics['ce_loss'] = ce_loss.item()
            diagnostics['action_loss'] = action_penalty.item()
            diagnostics['time_loss'] = time_penalty.item()
            
        return logits, loss, diagnostics
