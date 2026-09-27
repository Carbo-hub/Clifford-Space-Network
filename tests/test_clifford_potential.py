"""
Diagnostic Script: Testing Clifford-TENN with Rotor Frequency Scaling and Logit Temperature
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


class CliffordTENN_Geometric(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_neurons: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_neurons = num_neurons

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)

        self.in_proj = nn.Linear(d_model, num_neurons * 16)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        self.biv_proj = nn.Linear(d_model, num_neurons * 6)

        # 1. KEY FIX: Learnable Frequency/Rotor Multiplier (removes the 40x attenuation)
        self.omega_scale = nn.Parameter(torch.ones(num_neurons, 1) * 10.0)

        # 2. Inter-neuron Clifford Mixing
        self.W_mix = nn.Parameter(torch.randn(num_neurons, num_neurons) * (1.0 / math.sqrt(num_neurons)))

        self.ln_f = nn.LayerNorm(num_neurons * 16)
        self.head = nn.Linear(num_neurons * 16, vocab_size, bias=False)

        # 3. KEY FIX: Learnable Logit Temperature for Bounded Manifolds (ArcFace/CLIP style)
        self.logit_scale = nn.Parameter(torch.ones(1) * 5.0)

    def forward(self, idx: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        x_conv = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2))

        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 4, 4)
        gamma = F.softplus(self.gamma_proj(x_conv))
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1)

        # Bivectors with unconstrained angular frequency scale
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 6)
        biv = biv * self.omega_scale # Allows rotating across full phase circle [0, 2pi]

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
        
        # Logits with learnable temperature scaling
        logits = self.logit_scale * self.head(feat)
        return logits


def test_potential():
    train_data, val_data, vocab_size = get_data()
    model = CliffordTENN_Geometric(vocab_size=vocab_size, d_model=64, num_neurons=16).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    print("Training Geometric Clifford-TENN for 200 steps...")
    t0 = time.time()
    for step in range(1, 201):
        x, y = get_batch(train_data, batch_size=32, seq_len=128)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"  Step {step:3d}/200 | Loss: {loss.item():.4f} | Logit Scale: {model.logit_scale.item():.2f}")

    print(f"Finished in {time.time() - t0:.1f}s.")

    # Validation
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(val_data, batch_size=16, seq_len=128)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item() * y.numel()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc = 100.0 * total_correct / total_tokens
    bpc = avg_loss / math.log(2)

    print("\n--- RESULTS WITH GEOMETRIC TUNING ---")
    print(f"Perplexity (PPL):   {ppl:.2f}  (Was 10.37)")
    print(f"Top-1 Accuracy:     {acc:.2f}% (Was 33.29%)")
    print(f"Bits-Per-Char:      {bpc:.3f}  (Was 3.375)")
    print(f"Val Loss:           {avg_loss:.4f} (Was 2.3393)")

if __name__ == '__main__':
    test_potential()
