"""
Modular Model Architectures for 10M-100M Language Modeling on RTX 4090:
1. CSN-LM: Clifford Space Network with Spin(4)/Spin(8) Lie Rotors + Fused CUDA scan
2. ModernTransformerLM: Llama-3-style with RoPE + RMSNorm + SwiGLU + SDPA (FlashAttention-2)
3. MambaLM: Selective State Space Model (S6)
4. LiquidCfCLM: Closed-form Continuous-time Network with adaptive pacing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import CSN Rotor components
from csn.rotors import cayley_rotor_4x4, cayley_rotor_8x8
from csn.scan import parallel_rotor_scan


# ==============================================================================
# 1. Modern LLaMA-3 Style Transformer Baseline (RoPE + RMSNorm + SwiGLU + SDPA)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, L, H, D]
    L = x.shape[1]
    cos = cos[:L, None, :].to(x.device, dtype=x.dtype)
    sin = sin[:L, None, :].to(x.device, dtype=x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModernTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim)

        q = apply_rope(q, cos, sin).transpose(1, 2)  # [B, H, L, D_h]
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.0+ FlashAttention / SDPA kernel
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.o_proj(attn)

        x = x + self.mlp(self.norm2(x))
        return x


class ModernTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 1536,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        cos, sin = precompute_rope_freqs(d_model // num_heads, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ==============================================================================
# 2. Clifford Space Network (CSN-LM) for 4090 with Fused CUDA Acceleration
# ==============================================================================

class CSNLMBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, rotor_dim: int = 4, d_ff: int = 1536):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rotor_dim = rotor_dim
        
        self.norm1 = RMSNorm(d_model)
        self.proj_in = nn.Linear(d_model, num_heads * rotor_dim * rotor_dim, bias=False)
        self.proj_biv = nn.Linear(d_model, num_heads * ((rotor_dim * (rotor_dim - 1)) // 2), bias=False)
        self.gamma_log = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.proj_out = nn.Linear(num_heads * rotor_dim * rotor_dim, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

        triu_idx = torch.triu_indices(rotor_dim, rotor_dim, offset=1)
        self.register_buffer("triu_u", triu_idx[0], persistent=False)
        self.register_buffer("triu_v", triu_idx[1], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.rotor_dim

        h = self.norm1(x)
        C_inj = self.proj_in(h).view(B, L, H, D, D)

        biv = self.proj_biv(h).view(B, L, H, -1)
        Omega = torch.zeros(B, L, H, D, D, dtype=x.dtype, device=x.device)
        Omega[..., self.triu_u, self.triu_v] = biv
        Omega[..., self.triu_v, self.triu_u] = -biv
        Omega = Omega.clamp(-3.0, 3.0)

        if D == 4:
            Q = cayley_rotor_4x4(Omega, use_schulz=True)
        else:
            Q = cayley_rotor_8x8(Omega, use_schulz=True)

        gamma = F.softplus(self.gamma_log).view(1, 1, H, 1, 1)
        M = torch.sigmoid(-gamma) * Q

        X = parallel_rotor_scan(M, C_inj, is_spinor_bundle=True)
        x = x + self.proj_out(X.contiguous().view(B, L, -1))
        x = x + self.mlp(self.norm2(x))
        return x


class CSNLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        rotor_dim: int = 4,
        d_ff: int = 1536
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            CSNLMBlock(d_model, num_heads=num_heads, rotor_dim=rotor_dim, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ==============================================================================
# 3. Mamba S6 Baseline (Selective State Space Model)
# ==============================================================================

class MambaS6Block(nn.Module):
    """
    Selective State Space Block: h_t = A_t h_{t-1} + B_t x_t, y_t = C_t h_t
    Equipped with input-dependent Delta_t, B_t, C_t selection mechanics.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_ff: int = 1536):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm1 = RMSNorm(d_model)
        
        # SSM Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_state + d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Continuous S4D real diagonal transition parameter log(-A)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        xz = self.in_proj(h)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # 1D Depthwise Conv
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        # Selection projections: dt, B, C
        ssm_in = self.x_proj(x_act)
        delta_in, B_t, C_t = torch.split(ssm_in, [D, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_in))  # [B, L, D]

        # Discretization: A_bar = exp(-exp(A_log) * delta)
        A = -torch.exp(self.A_log)  # [D, N]
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
        dB = delta.unsqueeze(-1) * B_t.unsqueeze(2)                       # [B, L, D, N]
        dC = C_t.unsqueeze(2)                                             # [B, L, D, N]

        # Parallel Associative Scan for SSM
        u = x_act.unsqueeze(-1) * dB                                      # [B, L, D, N]
        
        # Prefix scan over L
        # (Using sequential log-space scan for robust baseline execution)
        y = torch.zeros_like(u)
        curr = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(L):
            curr = dA[:, t] * curr + u[:, t]
            y[:, t] = curr
        
        ssm_out = (y * dC).sum(dim=-1) + x_act * self.D
        y_out = ssm_out * F.silu(z_branch)
        x = x + self.out_proj(y_out)
        x = x + self.mlp(self.norm2(x))
        return x


class MambaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 8, d_state: int = 16, d_ff: int = 1536):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MambaS6Block(d_model, d_state=d_state, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


# ==============================================================================
# 4. Liquid AI Baseline (Closed-form Continuous CfC)
# ==============================================================================

class LiquidCfCBlock(nn.Module):
    """
    Closed-form Continuous-depth (CfC) recurrence:
        h_t = sigma(-f(x_t)*dt) * tanh(W1*x_t) + (1 - sigma(-f(x_t)*dt)) * tanh(W2*x_t)
    """
    def __init__(self, d_model: int, d_ff: int = 1536):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.w_decay = nn.Linear(d_model, d_model)
        self.w_gate1 = nn.Linear(d_model, d_model)
        self.w_gate2 = nn.Linear(d_model, d_model)
        self.w_rec = nn.Linear(d_model, d_model)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h_in = self.norm1(x)
        
        # Adaptive decay
        decay_rate = F.softplus(self.w_decay(h_in))
        cond = torch.sigmoid(-decay_rate)

        # Non-linear liquid transitions
        g1 = torch.tanh(self.w_gate1(h_in))
        g2 = torch.tanh(self.w_gate2(h_in))
        
        # Sequential continuous evolution
        out = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(L):
            state = cond[:, t] * torch.tanh(self.w_rec(state)) + (1.0 - cond[:, t]) * (g1[:, t] + g2[:, t])
            out[:, t] = state

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class LiquidLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 8, d_ff: int = 1536):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LiquidCfCBlock(d_model, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
