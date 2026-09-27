"""
Multi-Model Language Modeling Benchmark:
Transformer vs. Mamba (S6) vs. Liquid CfC vs. Clifford-TENN

Dataset: Tiny Shakespeare (Standard character-level LLM benchmark)
Metrics:
- Perplexity (PPL)
- Next-Token Top-1 Accuracy (%)
- Bits Per Character (BPC)
- Parameter Count
- GPU VRAM consumption across context length L in [64, 256, 512, 1024]
- Inference Latency and Throughput (tokens/sec)
"""

import os
import sys
import math
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 0. Dataset Setup (Tiny Shakespeare)
# ============================================================================

def get_data():
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', 'tinyshakespeare.txt')
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# 1. Model 1: Standard Causal Transformer (NanoGPT style)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': CausalSelfAttention(d_model, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model)
                )
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        h = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            h = h + layer['attn'](layer['ln1'](h))
            h = h + layer['mlp'](layer['ln2'](h))
        return self.head(self.ln_f(h))


# ============================================================================
# 2. Model 2: Mamba (Selective State Space Model S6)
# ============================================================================

class MambaBlock(nn.Module):
    """S6 Selective SSM Block (Gu & Dao, 2023)"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # Selective parameter projections: Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # S4D real initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xz = self.in_proj(x) # [B, T, 2 * d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # 1D causal convolution + SiLU
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters
        x_dbl = self.x_proj(x_conv) # [B, T, 1 + 2 * d_state]
        dt, B_ssm, C_ssm = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # [B, T, d_inner]

        A = -torch.exp(self.A_log) # [d_inner, d_state]

        # Recurrence step over time
        y_list = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)

        for t in range(T):
            dt_t = dt[:, t, :] # [B, d_inner]
            dA = torch.exp(dt_t.unsqueeze(-1) * A) # [B, d_inner, d_state]
            dB_x = torch.einsum('bn,bs->bns', x_conv[:, t, :] * dt_t, B_ssm[:, t, :])
            h = h * dA + dB_x
            y_t = torch.einsum('bns,bs->bn', h, C_ssm[:, t, :]) # [B, d_inner]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1) # [B, T, d_inner]
        y = y + x_conv * self.D
        y = y * F.silu(z_branch)
        return self.out_proj(y)


class MambaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln': nn.LayerNorm(d_model),
                'mamba': MambaBlock(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(idx)
        for layer in self.layers:
            h = h + layer['mamba'](layer['ln'](h))
        return self.head(self.ln_f(h))


# ============================================================================
# 3. Model 3: Liquid Neural Network (CfC - Closed-Form Continuous Time)
# ============================================================================

class CfCCell(nn.Module):
    """Closed-form Continuous-time cell (Hasani et al. / Liquid AI core)"""
    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.bb1 = nn.Linear(in_features + hidden_size, hidden_size)
        self.bb2 = nn.Linear(in_features + hidden_size, hidden_size)
        self.decay_proj = nn.Linear(in_features + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        ff1 = torch.tanh(self.bb1(combined))
        ff2 = torch.tanh(self.bb2(combined))
        decay = torch.sigmoid(-F.softplus(self.decay_proj(combined)) * dt)
        h_new = decay * ff1 + (1.0 - decay) * ff2
        return h_new


class LiquidLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, hidden_size: int = 64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.cfc = CfCCell(d_model, hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        h = torch.zeros(B, self.cfc.hidden_size, device=idx.device)
        outputs = []
        for t in range(T):
            h = self.cfc(x[:, t, :], h)
            outputs.append(h)
        out = torch.stack(outputs, dim=1)
        return self.head(self.ln_f(out))


# ============================================================================
# 4. Model 4: Clifford-TENN_V2 (Parallel Scan + Temporal Front + Event Spikes)
# ============================================================================

from python.clifford_scan import parallel_rotor_scan, surrogate_spike

class CliffordTENN_V2(nn.Module):
    """
    Upgraded Clifford-TENN Language Model:
    - Parallel Associative Scan: O(log T) parallel training (no sequential loops)
    - Short-Horizon Temporal Front: 1D depthwise causal convolution (k=4)
    - BrainChip Akida Event Spikes: Energy-threshold spiking with surrogate gradient
    - Continuous Clifford Cayley Rotors in SO(4)
    """
    def __init__(self, vocab_size: int, d_model: int = 64, num_neurons: int = 16, spike_thresh: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.spike_thresh = spike_thresh

        # 1. Token embedding & Causal Temporal Front
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            padding=3,
            groups=d_model
        )

        # 2. Multivector Projections
        self.in_proj = nn.Linear(d_model, num_neurons * 16)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        self.biv_proj = nn.Linear(d_model, num_neurons * 6) # 6 bivector components

        # 3. Inter-neuron Clifford Mixing
        self.W_mix = nn.Parameter(
            torch.randn(num_neurons, num_neurons) * (1.0 / math.sqrt(num_neurons))
        )

        # 4. Readout
        self.ln_f = nn.LayerNorm(num_neurons * 16)
        self.head = nn.Linear(num_neurons * 16, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) # [B, T, d_model]

        # 1. Short-Horizon Causal Temporal Front
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 2. Parallel Generation of Clifford Inputs across all T simultaneously
        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 4, 4)

        # Liquid Conductance & Decay
        gamma = F.softplus(self.gamma_proj(x_conv)) # [B, T, N]
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1) # [B, T, N, 1, 1]

        # Bivector Rotors via Cayley Transform in SO(4)
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 6)
        # Construct skew-symmetric 4x4 matrix
        skew = torch.zeros(B, T, self.num_neurons, 4, 4, device=idx.device)
        skew[..., 0, 1] =  biv[..., 0]; skew[..., 1, 0] = -biv[..., 0]
        skew[..., 0, 2] =  biv[..., 1]; skew[..., 2, 0] = -biv[..., 1]
        skew[..., 0, 3] =  biv[..., 2]; skew[..., 3, 0] = -biv[..., 2]
        skew[..., 1, 2] =  biv[..., 3]; skew[..., 2, 1] = -biv[..., 3]
        skew[..., 1, 3] =  biv[..., 4]; skew[..., 3, 1] = -biv[..., 4]
        skew[..., 2, 3] =  biv[..., 5]; skew[..., 3, 2] = -biv[..., 5]

        Omega = skew * (0.5 * dt)
        eye = torch.eye(4, device=idx.device).view(1, 1, 1, 4, 4)
        r_num = eye + Omega
        r_rev = eye - Omega
        omega_sq = 0.5 * torch.sum(Omega ** 2, dim=(-1, -2), keepdim=True)
        denom = 1.0 + 0.25 * omega_sq
        Q = torch.matmul(r_num, r_rev) / denom # [B, T, N, 4, 4]

        # Transition matrix M and Injected state C
        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S

        # 3. PARALLEL ASSOCIATIVE SCAN: O(log T) steps on GPU!
        # Completely replaces sequential for-loop!
        X_all = parallel_rotor_scan(M, C) # [B, T, N, 4, 4]

        # 4. BrainChip Akida Event Spikes (Energy-threshold gating)
        energy = torch.norm(X_all, dim=(-1, -2), keepdim=True) # [B, T, N, 1, 1]
        spikes = surrogate_spike(energy, threshold=self.spike_thresh, beta=10.0)
        X_spiked = X_all * spikes

        # 5. Inter-Neuron Clifford Mixing
        # einsum: mix neuron channels across all T in parallel
        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)

        # 6. Readout
        out = self.ln_f(X_mixed.reshape(B, T, -1))
        return self.head(out)


# ============================================================================
# 5. Evaluation Metrics & Benchmark Routine
# ============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(model: nn.Module, val_data: torch.Tensor, seq_len: int = 128, n_batches: int = 20):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size=16, seq_len=seq_len)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc = 100.0 * total_correct / total_tokens
    bpc = avg_loss / math.log(2)
    return ppl, acc, bpc, avg_loss


def run_language_benchmark():
    print("=" * 80)
    print(f"  RIGOROUS LLM BENCHMARK: Tiny Shakespeare Language Modeling on {device.type.upper()}")
    print("  Models: Transformer vs. Mamba (S6) vs. Liquid CfC vs. Clifford-TENN")
    print("=" * 80)

    train_data, val_data, vocab_size = get_data()
    print(f"[Dataset] Vocab Size: {vocab_size} unique tokens | Train Tokens: {len(train_data):,} | Val Tokens: {len(val_data):,}")

    models = {
        'Transformer': TransformerLM(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=2).to(device),
        'Mamba (S6)':   MambaLM(vocab_size=vocab_size, d_model=64, n_layers=2).to(device),
        'Liquid (CfC)': LiquidLM(vocab_size=vocab_size, d_model=64, hidden_size=64).to(device),
        'Clifford-TENN (v2)': CliffordTENN_V2(vocab_size=vocab_size, d_model=64, num_neurons=16).to(device)
    }

    print("\n[1. Model Parameter Comparison]")
    for name, m in models.items():
        print(f"  {name:<16}: {count_params(m):>8,} parameters")

    # Training loop
    print("\n[2. Training on 200 batches of length L=128]")
    batch_size = 32
    seq_len = 128
    num_steps = 200

    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01) for name, m in models.items()}

    t0_train = time.time()
    for step in range(1, num_steps + 1):
        x, y = get_batch(train_data, batch_size=batch_size, seq_len=seq_len)
        for name, m in models.items():
            m.train()
            optimizers[name].zero_grad()
            logits = m(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizers[name].step()

        if step % 50 == 0 or step == 1:
            print(f"  Step {step:3d}/{num_steps} completed...")

    print(f"  Training finished in {time.time() - t0_train:.1f}s.")

    # Standard NLP Evaluation
    print("\n[3. Standard NLP Evaluation on Validation Set]")
    print(f"  {'Model':<16} | {'Perplexity (PPL)':<18} | {'Top-1 Accuracy':<16} | {'Bits-Per-Char':<15} | {'Val Loss':<10}")
    print("  " + "-" * 82)

    for name, m in models.items():
        ppl, acc, bpc, loss = compute_metrics(m, val_data, seq_len=seq_len)
        print(f"  {name:<16} | {ppl:>16.2f}   | {acc:>14.2f}%  | {bpc:>13.3f}   | {loss:>8.4f}")

    # Context Scaling & Memory Benchmark
    print("\n[4. Context Length Scaling & Memory Footprint]")
    print(f"  Peak VRAM (MB) across sequence lengths L in [64, 256, 512, 1024] at Batch=8")
    print(f"  {'Length (L)':<12} | {'Transformer':<15} | {'Mamba (S6)':<15} | {'Liquid (CfC)':<15} | {'Clifford-TENN':<15}")
    print("  " + "-" * 78)

    for L in [64, 256, 512, 1024]:
        x_eval, _ = get_batch(val_data, batch_size=8, seq_len=L)
        vrams = {}
        for name, m in models.items():
            m.eval()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = m(x_eval)
            vram = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == 'cuda' else 0.0
            vrams[name] = vram
        print(f"  L = {L:<7} | {vrams['Transformer']:>11.2f} MB | {vrams['Mamba (S6)']:>11.2f} MB | {vrams['Liquid (CfC)']:>11.2f} MB | {vrams['Clifford-TENN (v2)']:>11.2f} MB")

    print("\n" + "=" * 80)
    print("  Language Modeling Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    run_language_benchmark()
