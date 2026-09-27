"""
Multi-Head Clifford-TENN Cl(6, 0): Geometric Subspace Factorization
with Per-Head Adaptive Continuous-Time Warping and Exact Cayley Rotors on M_8(R).

Mathematical Architecture:
1. Subspace Factorization: Lie group direct product Spin(6)^H = (SU(4))^H.
2. Per-Head Multi-Scale Time Pacing:
       Delta t^{(h)}_t = softplus(W_dt^{(h)} x_t + b_dt^{(h)}) * dt_base
   allows heads to autonomously specialize into distinct frequency/timescale bands
   (fast phonetic n-grams vs. slow syntactic/semantic tracking).
3. Batched Cayley Rotors on M_8(R) (64 state dimensions per neuron).
4. Zero-Copy Parallel Associative Scan across all heads simultaneously via batch-folding.
5. Per-head Neuromorphic Event Spiking and Block-Diagonal Synaptic Mixing.
6. Geometric Gated Linear Unit (GGLU) & Canonical Riemannian Geodesic Head.
"""

import math
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.clifford_scan import parallel_rotor_scan, surrogate_spike
from python.clifford_model import RiemannianGeodesicHead, GeometricGatedLinearUnit
from python.clifford_cl6 import cayley_rotor_8x8


class MultiHeadAdaptiveTime(nn.Module):
    """
    Per-Head Adaptive Continuous-Time Pacing.
    Predicts independent time steps Delta t^{(h)}_t for each geometric head:
        Delta t^{(h)}_t = softplus(W_dt^{(h)} x_t + b_dt^{(h)}) * dt_base
    """
    def __init__(self, d_model: int, num_heads: int, dt_base: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.dt_base = dt_base
        self.proj = nn.Linear(d_model, num_heads)
        
        # Initialize W_dt near zero and b_dt such that softplus(b_dt) = 1.0
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, math.log(math.e - 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            dt: [B, T, H] strictly positive time steps
        """
        scale = F.softplus(self.proj(x)) # [B, T, H]
        dt = scale * self.dt_base
        return dt


class MultiHeadCliffordCL6Block(nn.Module):
    """
    Multi-Head Monolithic Cl(6, 0) Layer with Factorized Geometric Subspaces.
    
    Total multivector capacity = num_heads * num_neurons_per_head * 64 dimensions.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 3,
        num_neurons_per_head: int = 2,
        kernel_size: int = 4,
        dt_base: float = 0.05,
        spike_threshold: float = 0.1,
        spike_beta: float = 10.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_neurons_per_head = num_neurons_per_head
        self.total_neurons = num_heads * num_neurons_per_head
        self.dt_base = dt_base
        self.spike_threshold = spike_threshold
        self.spike_beta = spike_beta
        
        # 1. Pre-Normalization
        self.ln_1 = nn.LayerNorm(d_model)
        
        # 2. Causal 1D Convolution Front
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        
        # 3. Per-Head Adaptive Time Pacing Head
        self.time_head = MultiHeadAdaptiveTime(d_model, num_heads=num_heads, dt_base=dt_base)
        
        # 4. Multi-Head Clifford Projections in Cl(6, 0)
        # 64 components per neuron (8x8 real matrix)
        self.in_proj = nn.Linear(d_model, self.total_neurons * 64)
        self.gamma_proj = nn.Linear(d_model, self.total_neurons)
        # 28 bivector generators per neuron for so(8) / Spin(6)
        self.biv_proj = nn.Linear(d_model, self.total_neurons * 28)
        
        # Frequency scale per head and neuron
        self.omega_scale = nn.Parameter(torch.ones(num_heads, num_neurons_per_head, 1) * 10.0)
        
        # 5. Per-Head Block-Diagonal Synaptic Mixing
        # Mixing occurs internally within each head: [H, N_head, N_head]
        self.W_mix = nn.Parameter(
            torch.randn(num_heads, num_neurons_per_head, num_neurons_per_head) / math.sqrt(num_neurons_per_head)
        )
        self.out_proj = nn.Linear(self.total_neurons * 64, d_model)
        
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
            Omega: [B, T, H, N_head, 8, 8] skew-symmetric generator tensor
            dt: [B, T, H] learned per-head adaptive time steps
        """
        B, T, D = x.shape
        H = self.num_heads
        Nh = self.num_neurons_per_head
        residual = x
        
        x_norm = self.ln_1(x)
        x_conv = F.silu(self.conv1d(x_norm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        
        # 1. Predict Per-Head Adaptive Time Steps: [B, T, H]
        dt = self.time_head(x_conv)
        # Reshape dt for broadcast: [B, T, H, 1, 1, 1]
        dt_exp = dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 2. Injected multivector field S: [B, T, H, Nh, 8, 8]
        S = self.in_proj(x_conv).view(B, T, H, Nh, 8, 8)
        
        # 3. Liquid Decay per neuron: [B, T, H, Nh, 1, 1]
        gamma = F.softplus(self.gamma_proj(x_conv)).view(B, T, H, Nh, 1, 1)
        decay = torch.sigmoid(-gamma * dt_exp)
        
        # 4. Bivector Generators & Cayley Rotor in Spin(6)
        biv = self.biv_proj(x_conv).view(B, T, H, Nh, 28) * self.omega_scale
        skew = torch.zeros(B, T, H, Nh, 8, 8, device=x.device)
        skew[..., self.triu_u, self.triu_v] = biv
        skew[..., self.triu_v, self.triu_u] = -biv
        
        Omega = skew * (0.5 * dt_exp)
        
        # 5. Efficient Batch-Folding for Parallel Scan
        # Fold batch and heads: [B, T, H, Nh, 8, 8] -> [B * H, T, Nh, 8, 8]
        B_prime = B * H
        Omega_flat = Omega.transpose(1, 2).reshape(B_prime, T, Nh, 8, 8)
        S_flat = S.transpose(1, 2).reshape(B_prime, T, Nh, 8, 8)
        decay_flat = decay.transpose(1, 2).reshape(B_prime, T, Nh, 1, 1)
        
        # Exact Cayley Rotor Q: (I - Omega)^(-1) (I + Omega)
        Q_flat = cayley_rotor_8x8(Omega_flat)
        
        # Scan operators
        M_flat = torch.sqrt(decay_flat) * Q_flat
        C_flat = (1.0 - decay_flat) * S_flat
        
        # Vectorized O(log T) Parallel Associative Scan across all heads simultaneously
        X_all_flat = parallel_rotor_scan(M_flat, C_flat)
        
        # 6. Neuromorphic Event Spiking
        energy = torch.norm(X_all_flat, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=self.spike_threshold, beta=self.spike_beta)
        X_spiked_flat = X_all_flat * spikes
        
        # 7. Unfold Heads & Inter-neuron Mixing within each head
        # [B * H, T, Nh, 8, 8] -> [B, H, T, Nh, 8, 8] -> [B, T, H, Nh, 8, 8]
        X_unflat = X_spiked_flat.reshape(B, H, T, Nh, 8, 8).transpose(1, 2)
        
        # Block-diagonal mixing: W_mix [H, Nh, Nh] mixes neurons within each head
        X_mixed = torch.einsum('hmn,bthnrc->bthmrc', self.W_mix, X_unflat)
        
        # 8. Concatenate all heads into single representation
        feat = X_mixed.reshape(B, T, self.total_neurons * 64)
        
        # 9. Output projection & GGLU
        h = residual + self.out_proj(feat)
        out = h + self.gglu(self.ln_2(h))
        
        return out, Omega, dt


class MultiHeadCliffordLM(nn.Module):
    """
    Complete Multi-Head Clifford-TENN Causal Language Model.
    Features:
    - Multi-Head Cl(6, 0) Monolithic Blocks
    - Per-Head Adaptive Time Pacing (Multi-timescale Neural ODE)
    - Canonical Riemannian Geodesic Head with Bayesian Prior Calibration
    - Combined Hamiltonian Action + Temporal Smoothness Regularization
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        num_layers: int = 2,
        num_heads: int = 3,
        num_neurons_per_head: int = 2,
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
        self.num_heads = num_heads
        self.num_neurons_per_head = num_neurons_per_head
        self.dt_base = dt_base
        self.lambda_kin = lambda_kin
        self.lambda_smooth = lambda_smooth
        self.lambda_time = lambda_time
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            MultiHeadCliffordCL6Block(
                d_model=d_model,
                num_heads=num_heads,
                num_neurons_per_head=num_neurons_per_head,
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
            loss: scalar loss
            diagnostics: dictionary with per-head dt statistics, action loss, and time loss
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        total_e_kin = 0.0
        total_e_smooth = 0.0
        total_time_loss = 0.0
        all_dts = [] # per block [B, T, H]
        
        for block in self.blocks:
            x, Omega, dt = block(x)
            all_dts.append(dt)
            
            # Hamiltonian Least-Action Regularization
            e_kin = (Omega ** 2).mean()
            if T > 1:
                e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
                time_smooth = (((dt[:, 1:] - dt[:, :-1]) / self.dt_base) ** 2).mean()
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
        
        # Compute diagnostics for each head across all layers
        stacked_dt = torch.stack(all_dts) # [L, B, T, H]
        for h in range(self.num_heads):
            dt_h = stacked_dt[..., h]
            diagnostics[f'head_{h}_mean_dt'] = dt_h.mean().item()
            diagnostics[f'head_{h}_min_dt'] = dt_h.min().item()
            diagnostics[f'head_{h}_max_dt'] = dt_h.max().item()
            
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            action_penalty = self.lambda_kin * total_e_kin + self.lambda_smooth * total_e_smooth
            time_penalty = self.lambda_time * total_time_loss
            loss = ce_loss + action_penalty + time_penalty
            
            diagnostics['ce_loss'] = ce_loss.item()
            diagnostics['action_loss'] = action_penalty.item()
            diagnostics['time_loss'] = time_penalty.item()
            
        return logits, loss, diagnostics
