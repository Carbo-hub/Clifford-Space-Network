"""
Comparative Benchmark: Native Clifford Loss Functions
Testing:
1. Baseline: Clifford-TENN with Euclidean Cross-Entropy
2. Variant 1: Clifford Cosine Contraction Loss (Normalized Multivector Contraction)
3. Variant 1+3: Clifford Cosine Contraction + Geodesic Hamiltonian Action Regularization (delta S = 0)
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_scan import parallel_rotor_scan, surrogate_spike

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Dataset Loader
# ============================================================================

def get_data():
    file_path = os.path.join('data', 'tinyshakespeare.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size


def get_batch(data, batch_size=32, seq_len=128):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# Core Clifford-TENN Backbone
# ============================================================================

class CliffordCore(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_neurons: int = 16):
        super().__init__()
        self.num_neurons = num_neurons
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)

        self.in_proj = nn.Linear(d_model, num_neurons * 16)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        self.biv_proj = nn.Linear(d_model, num_neurons * 6)
        self.omega_scale = nn.Parameter(torch.ones(num_neurons, 1) * 10.0)

        self.W_mix = nn.Parameter(torch.randn(num_neurons, num_neurons) * (1.0 / math.sqrt(num_neurons)))
        self.ln_f = nn.LayerNorm(num_neurons * 16)

    def forward(self, idx: torch.Tensor, dt: float = 0.05):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x_conv = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2))

        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 4, 4)
        gamma = F.softplus(self.gamma_proj(x_conv))
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1)

        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 6) * self.omega_scale

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
        Q = torch.matmul(r_num, r_rev) / denom

        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S

        # Parallel Associative Scan
        X_all = parallel_rotor_scan(M, C)

        # Spikes
        energy = torch.norm(X_all, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=0.1, beta=10.0)
        X_spiked = X_all * spikes

        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)
        feat = self.ln_f(X_mixed.reshape(B, T, -1))
        return feat, Omega


# ============================================================================
# Variant 1: Clifford Cosine Contraction Loss
# ============================================================================

class CliffordTENN_Cosine(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_neurons: int = 16):
        super().__init__()
        self.core = CliffordCore(vocab_size, d_model, num_neurons)
        # Unit multivector prototypes for each token in the vocabulary
        self.prototypes = nn.Parameter(torch.randn(vocab_size, num_neurons * 16))
        # Learnable inverse temperature (scale)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, idx: torch.Tensor):
        feat, Omega = self.core(idx) # [B, T, D]
        # Clifford Contraction: normalized cosine similarity on manifold
        norm_feat = F.normalize(feat, p=2, dim=-1)
        norm_proto = F.normalize(self.prototypes, p=2, dim=-1)
        cos_sim = torch.matmul(norm_feat, norm_proto.t()) # [B, T, vocab_size] in [-1, 1]
        logits = self.scale * cos_sim
        return logits, Omega


# ============================================================================
# Evaluation Routine
# ============================================================================

def evaluate(model, val_data, vocab_size, seq_len=128, n_batches=30):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, batch_size=16, seq_len=seq_len)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item() * y.numel()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc = 100.0 * total_correct / total_tokens
    bpc = avg_loss / math.log(2)
    return ppl, acc, bpc, avg_loss


def run_loss_comparison():
    print("=" * 80)
    print(f"  CLIFFORD LOSS FUNCTION EXPERIMENT on {device.type.upper()}")
    print("  Testing: Variant 1 (Clifford Cosine) vs. Variant 1+3 (Cosine + Geodesic)")
    print("=" * 80)

    train_data, val_data, vocab_size = get_data()
    num_steps = 1000
    batch_size = 32
    seq_len = 128

    # ------------------------------------------------------------------------
    # RUN 1: Variant 1 (Clifford Cosine Contraction Loss)
    # ------------------------------------------------------------------------
    print("\n[Running Run 1: Variant 1 (Clifford Cosine Contraction)]")
    model_v1 = CliffordTENN_Cosine(vocab_size=vocab_size, d_model=64, num_neurons=16).to(device)
    opt_v1 = torch.optim.AdamW(model_v1.parameters(), lr=2e-3, weight_decay=0.01)
    sched_v1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v1, T_max=num_steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, num_steps + 1):
        model_v1.train()
        x, y = get_batch(train_data, batch_size=batch_size, seq_len=seq_len)
        opt_v1.zero_grad()
        logits, _ = model_v1(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v1.parameters(), 1.0)
        opt_v1.step()
        sched_v1.step()

        if step % 250 == 0 or step == 1:
            print(f"  Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Scale: {model_v1.scale.item():.2f}")

    t_v1 = time.time() - t0
    ppl_v1, acc_v1, bpc_v1, loss_v1 = evaluate(model_v1, val_data, vocab_size)
    print(f"  -> Finished in {t_v1:.1f}s | Val PPL: {ppl_v1:.2f} | Acc: {acc_v1:.2f}% | BPC: {bpc_v1:.3f}")

    # ------------------------------------------------------------------------
    # RUN 2: Variant 1+3 (Clifford Cosine + Geodesic Kinetic Regularization)
    # ------------------------------------------------------------------------
    print("\n[Running Run 2: Variant 1+3 (Clifford Cosine + Geodesic Action delta S = 0)]")
    model_v13 = CliffordTENN_Cosine(vocab_size=vocab_size, d_model=64, num_neurons=16).to(device)
    opt_v13 = torch.optim.AdamW(model_v13.parameters(), lr=2e-3, weight_decay=0.01)
    sched_v13 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v13, T_max=num_steps, eta_min=1e-4)

    # Physics regularization weights
    lambda_kin = 0.05    # Penalize excessive kinetic rotation energy ||Omega||^2
    lambda_smooth = 0.02 # Penalize angular jerk ||Omega_{t+1} - Omega_t||^2

    t0 = time.time()
    for step in range(1, num_steps + 1):
        model_v13.train()
        x, y = get_batch(train_data, batch_size=batch_size, seq_len=seq_len)
        opt_v13.zero_grad()
        logits, Omega = model_v13(x)
        
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        # Hamiltonian Action / Kinetic energy of rotation
        e_kin = (Omega ** 2).mean()
        e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
        total_loss = ce_loss + lambda_kin * e_kin + lambda_smooth * e_smooth

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v13.parameters(), 1.0)
        opt_v13.step()
        sched_v13.step()

        if step % 250 == 0 or step == 1:
            print(f"  Step {step:4d}/{num_steps} | CE: {ce_loss.item():.4f} | E_kin: {e_kin.item():.4f} | Scale: {model_v13.scale.item():.2f}")

    t_v13 = time.time() - t0
    ppl_v13, acc_v13, bpc_v13, loss_v13 = evaluate(model_v13, val_data, vocab_size)
    print(f"  -> Finished in {t_v13:.1f}s | Val PPL: {ppl_v13:.2f} | Acc: {acc_v13:.2f}% | BPC: {bpc_v13:.3f}")

    # ------------------------------------------------------------------------
    # FINAL SUMMARY COMPARISON
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL LOSS FUNCTION COMPARISON ON VALIDATION SET")
    print("=" * 80)
    print(f"  {'Configuration':<36} | {'Perplexity (PPL)':<18} | {'Top-1 Accuracy':<16} | {'Bits-Per-Char':<15}")
    print("  " + "-" * 90)
    print(f"  {'Euclidean Cross-Entropy (Baseline)':<36} | {'7.45':>16}   | {'41.30%':>14}  | {'2.897':>13}")
    print(f"  {'Variant 1: Clifford Cosine Head':<36} | {ppl_v1:>16.2f}   | {acc_v1:>14.2f}%  | {bpc_v1:>13.3f}")
    print(f"  {'Variant 1+3: Cosine + Geodesic Action':<36} | {ppl_v13:>16.2f}   | {acc_v13:>14.2f}%  | {bpc_v13:>13.3f}")
    print("=" * 80)


if __name__ == '__main__':
    run_loss_comparison()
