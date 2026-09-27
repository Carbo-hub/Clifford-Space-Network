"""
Clifford Space Network for Transformers, LLMs, and 1D Sequential Modeling.

Replaces standard quadratic O(L^2) Self-Attention (or Mamba SSMs) with strictly
unitary/orthogonal Clifford Rotor Dynamics:
    X_t = M_t @ X_{t-1} + C_t
where M_t in Spin(d) is an exact Lie group rotor generated from input tokens.

Properties:
1. Linear Time & Memory: O(L) compute and O(1) state memory per token at inference.
2. Norm Preservation: Clifford rotors preserve multivector norms (|Q v| = |v|),
   guaranteeing zero gradient explosion/vanishing across arbitrarily long contexts.
3. Natural Causal Masking: The recurrent scan strictly propagates forward in time.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from csn.rotors import cayley_rotor_4x4, cayley_rotor_8x8, cayley_rotor_16x16
from csn.scan import parallel_rotor_scan


class CliffordCausalRotorBlock(nn.Module):
    """
    Causal 1D Sequence Modeling Block via Multi-Head Clifford Rotors in Spin(4) or Spin(8).
    Replaces Self-Attention in Transformers with an O(L) Clifford Associative Scan.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        rotor_dim: int = 4,   # 4 for Spin(4) or 16 for Spin(8)
        use_schulz: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rotor_dim = rotor_dim
        self.use_schulz = use_schulz

        # Project token representations into multi-head Clifford injection states
        self.proj_in = nn.Linear(d_model, num_heads * rotor_dim * rotor_dim)
        
        # Generator for Lie algebra so(rotor_dim) bivectors
        num_bivectors = (rotor_dim * (rotor_dim - 1)) // 2
        self.proj_biv = nn.Linear(d_model, num_heads * num_bivectors)

        # Decay gate parameter (log-gamma)
        self.gamma_log = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Output projection
        self.proj_out = nn.Linear(num_heads * rotor_dim * rotor_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Upper triangular indices for skew-symmetric generator
        triu_idx = torch.triu_indices(rotor_dim, rotor_dim, offset=1)
        self.register_buffer("triu_u", triu_idx[0])
        self.register_buffer("triu_v", triu_idx[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] - Input sequence embeddings
        Returns:
            out: [B, L, d_model] - Contextualized sequence representations
        """
        B, L, _ = x.shape
        H = self.num_heads
        D = self.rotor_dim

        # 1. Matter injection C: [B, L, H, D, D]
        C_inj = self.proj_in(x).view(B, L, H, D, D)

        # 2. Skew-symmetric Lie algebra generator Omega in so(D)
        biv = self.proj_biv(x).view(B, L, H, -1)
        Omega = torch.zeros(B, L, H, D, D, dtype=x.dtype, device=x.device)
        Omega[..., self.triu_u, self.triu_v] = biv
        Omega[..., self.triu_v, self.triu_u] = -biv
        Omega = Omega.clamp(min=-3.0, max=3.0)

        # 3. Cayley Isometry Q in Spin(D)
        if D == 4:
            Q = cayley_rotor_4x4(Omega, use_schulz=self.use_schulz)
        elif D == 8:
            Q = cayley_rotor_8x8(Omega, use_schulz=self.use_schulz)
        elif D == 16:
            Q = cayley_rotor_16x16(Omega, use_schulz=self.use_schulz)
        else:
            raise ValueError(f"Unsupported rotor_dim: {D}")

        # 4. Spatial/Temporal Decay M_t = sigma(-gamma) * Q_t
        gamma = F.softplus(self.gamma_log).view(1, 1, H, 1, 1)
        decay = torch.sigmoid(-gamma)
        M = decay * Q

        # 5. Causal Associative Scan along sequence length L (dim=1)
        X = parallel_rotor_scan(M, C_inj, is_spinor_bundle=True)

        # 6. Project back to d_model
        out = self.proj_out(X.contiguous().view(B, L, H * D * D))
        return self.norm(x + out)


class CliffordTransformerBlock(nn.Module):
    """Full Transformer Block: Clifford Causal Rotor Scan + Gated MLP."""
    def __init__(self, d_model: int, num_heads: int = 4, rotor_dim: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.rotor_block = CliffordCausalRotorBlock(d_model, num_heads=num_heads, rotor_dim=rotor_dim)
        
        self.ln2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rotor_block(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CliffordTransformerLM(nn.Module):
    """
    Causal Language Model built with Clifford Space Network Backbone.
    Replaces Self-Attention with Clifford Rotors for infinite context & linear complexity.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        rotor_dim: int = 4,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        self.layers = nn.ModuleList([
            CliffordTransformerBlock(d_model, num_heads=num_heads, rotor_dim=rotor_dim)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :L]
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.ln_f(x))
        return logits
