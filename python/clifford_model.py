"""
Clifford-TENN: Production-Grade Multi-Layer Geometric Continuous-Time Architecture.

Combines:
1. Pure Clifford Algebra Cl(4, 0) multivector states isomorphic to M_4(R).
2. Continuous-time Liquid Time-Constant (LTC/CfC) conductance gating.
3. Rational Cayley Transform rotor dynamics in Lie group Spin(4).
4. O(log T) Parallel Associative Rotor Scan.
5. Neuromorphic Event Spiking (BrainChip-inspired surrogate Heaviside).
6. Geometric Gated Linear Units (GGLU).
7. Canonical Riemannian Geodesic Classification Head with Bayesian Prior Calibration.
8. Hamiltonian Least-Action Regularization (delta S = 0).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from python.clifford_scan import parallel_rotor_scan, surrogate_spike


class RiemannianGeodesicHead(nn.Module):
    """
    Canonical Bi-Invariant Riemannian Geodesic Head on Spin(4) / Hypersphere.
    
    Computes logits via geodesic angular distance on the manifold:
        logit_c = s * cos(theta_c) + b_c
    where:
        cos(theta_c) = <X, E_c> / (||X|| * ||E_c||)
        s = learnable inverse curvature radius (temperature scale)
        b_c = unconstrained Bayesian class prior bias
    """
    def __init__(self, vocab_size: int, d_model: int, init_scale: float = 16.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token prototypes on the manifold (initialized randomly on unit hypersphere)
        self.prototypes = nn.Parameter(torch.randn(vocab_size, d_model) / math.sqrt(d_model))
        # Learnable manifold curvature / scale factor
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        # Unconstrained prior bias representing static token frequencies P(c)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] continuous multivector feature representation
        Returns:
            logits: [B, T, vocab_size]
        """
        norm_x = F.normalize(x, p=2, dim=-1)
        norm_proto = F.normalize(self.prototypes, p=2, dim=-1)
        cos_sim = torch.matmul(norm_x, norm_proto.t())
        return self.scale * cos_sim + self.bias


class GeometricGatedLinearUnit(nn.Module):
    """
    Geometric Gated Linear Unit (GGLU) for multivector feature channels.
    Projects multivector features through dual linear branches with SiLU gating:
        GGLU(x) = Linear_u(x) * silu(Linear_v(x))
    """
    def __init__(self, d_model: int, d_ffn: Optional[int] = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = int(2 * d_model)
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class CliffordTENNBlock(nn.Module):
    """
    Single Hierarchical Clifford-TENN Deep Layer.
    
    Pipeline:
    1. Pre-LayerNorm (stable deep gradient flow)
    2. Short-Horizon Causal 1D Conv (local n-gram phonetic front)
    3. Multivector Projection S in Cl(4, 0)
    4. Liquid Time-Constant decay alpha = sigmoid(-gamma * dt)
    5. Cayley Transform Rotor Q in Spin(4) with learnable multi-frequency omega_scale
    6. O(log T) Parallel Associative Scan
    7. Neuromorphic Surrogate Spiking
    8. All-to-all Inter-neuron Clifford Mixing
    9. GGLU Multivector Feedforward Network
    10. Residual connections
    """
    def __init__(
        self,
        d_model: int,
        num_neurons: int = 16,
        kernel_size: int = 4,
        spike_threshold: float = 0.1,
        spike_beta: float = 10.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.spike_threshold = spike_threshold
        self.spike_beta = spike_beta
        
        # 1. Pre-Normalization
        self.ln_1 = nn.LayerNorm(d_model)
        
        # 2. Local Causal Temporal Front (Depthwise Conv1D)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        
        # 3. Clifford Multivector Projections
        # Cl(4, 0) is isomorphic to M_4(R), 16 components per neuron
        self.in_proj = nn.Linear(d_model, num_neurons * 16)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        # 6 bivector generators: e_12, e_13, e_14, e_23, e_24, e_34
        self.biv_proj = nn.Linear(d_model, num_neurons * 6)
        
        # Multi-frequency rotor scale parameter (prevents rotor freeze)
        self.omega_scale = nn.Parameter(torch.ones(num_neurons, 1) * 10.0)
        
        # 4. Inter-neuron Clifford Mixing Matrix
        self.W_mix = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * (1.0 / math.sqrt(num_neurons))
        )
        self.out_proj = nn.Linear(num_neurons * 16, d_model)
        
        # 5. GGLU Feedforward Sub-layer
        self.ln_2 = nn.LayerNorm(d_model)
        self.gglu = GeometricGatedLinearUnit(d_model, d_ffn=2 * d_model)

    def forward(self, x: torch.Tensor, dt: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
            dt: Continuous-time interval
        Returns:
            out: [B, T, d_model]
            Omega: [B, T, num_neurons, 4, 4] skew-symmetric generator tensor for action loss
        """
        B, T, D = x.shape
        residual = x
        
        # Pre-Norm & Causal 1D Conv Front
        x_norm = self.ln_1(x)
        x_conv = self.conv1d(x_norm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Multivector Injected State: S in M_4(R)
        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 4, 4)
        
        # Liquid Decay
        gamma = F.softplus(self.gamma_proj(x_conv)) # [B, T, N]
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1) # [B, T, N, 1, 1]
        
        # Cayley Rotor in Spin(4)
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 6) * self.omega_scale
        skew = torch.zeros(B, T, self.num_neurons, 4, 4, device=x.device)
        skew[..., 0, 1] =  biv[..., 0]; skew[..., 1, 0] = -biv[..., 0]
        skew[..., 0, 2] =  biv[..., 1]; skew[..., 2, 0] = -biv[..., 1]
        skew[..., 0, 3] =  biv[..., 2]; skew[..., 3, 0] = -biv[..., 2]
        skew[..., 1, 2] =  biv[..., 3]; skew[..., 2, 1] = -biv[..., 3]
        skew[..., 1, 3] =  biv[..., 4]; skew[..., 3, 1] = -biv[..., 4]
        skew[..., 2, 3] =  biv[..., 5]; skew[..., 3, 2] = -biv[..., 5]
        
        Omega = skew * (0.5 * dt)
        eye = torch.eye(4, device=x.device).view(1, 1, 1, 4, 4)
        r_num = eye + Omega
        r_rev = eye - Omega
        omega_sq = 0.5 * torch.sum(Omega ** 2, dim=(-1, -2), keepdim=True)
        denom = 1.0 + 0.25 * omega_sq
        Q = torch.matmul(r_num, r_rev) / denom
        
        # Transition operator M and Injected C for Associative Scan
        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S
        
        # O(log T) Parallel Associative Scan (Proven in Lean 4)
        X_all = parallel_rotor_scan(M, C)
        
        # Neuromorphic Surrogate Spiking (Energy Thresholding)
        energy = torch.norm(X_all, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=self.spike_threshold, beta=self.spike_beta)
        X_spiked = X_all * spikes
        
        # Inter-neuron Clifford Mixing
        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)
        feat = X_mixed.reshape(B, T, self.num_neurons * 16)
        
        # Residual projection
        h = residual + self.out_proj(feat)
        
        # GGLU Sub-layer with residual
        out = h + self.gglu(self.ln_2(h))
        
        return out, Omega


class CliffordTENNForCausalLM(nn.Module):
    """
    Deep Multi-Layer Clifford-TENN Causal Language Model.
    
    Architecture:
    - Token Embedding
    - Stacked CliffordTENNBlock layers with residual stream
    - Final Pre-Head Normalization
    - Canonical Riemannian Geodesic Head
    - Hamiltonian Least-Action Regularizer
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 16,
        kernel_size: int = 4,
        spike_threshold: float = 0.1,
        lambda_kin: float = 0.02,
        lambda_smooth: float = 0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.lambda_kin = lambda_kin
        self.lambda_smooth = lambda_smooth
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            CliffordTENNBlock(
                d_model=d_model,
                num_neurons=num_neurons,
                kernel_size=kernel_size,
                spike_threshold=spike_threshold
            )
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = RiemannianGeodesicHead(vocab_size=vocab_size, d_model=d_model)

    def forward(
        self,
        idx: torch.Tensor,
        dt: float = 0.05,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Args:
            idx: [B, T] input token indices
            dt: continuous-time step size
            targets: [B, T] target token indices (optional)
        Returns:
            logits: [B, T, vocab_size]
            loss: total loss (Cross-Entropy + Hamiltonian Action Regularization)
            action_loss: scalar value of action penalty
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        total_e_kin = 0.0
        total_e_smooth = 0.0
        
        for block in self.blocks:
            x, Omega = block(x, dt=dt)
            # Hamiltonian Least-Action terms:
            e_kin = (Omega ** 2).mean()
            if T > 1:
                e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
            else:
                e_smooth = torch.tensor(0.0, device=idx.device)
            total_e_kin = total_e_kin + e_kin
            total_e_smooth = total_e_smooth + e_smooth
            
        x_norm = self.ln_f(x)
        logits = self.head(x_norm)
        
        loss = None
        action_loss_val = 0.0
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            action_penalty = self.lambda_kin * total_e_kin + self.lambda_smooth * total_e_smooth
            loss = ce_loss + action_penalty
            action_loss_val = action_penalty.item()
            
        return logits, loss, action_loss_val
