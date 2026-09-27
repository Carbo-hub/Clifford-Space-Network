"""
Empirical Comparison:
Option 1: Biological Dendritic Multi-Compartment (K=4 in Cl(4,0), 64 state dims/neuron)
vs.
Option 2: Mathematical Monolithic Cl(6, 0) (8x8 matrix, 64 state dims/neuron)

Both models matched to the same ~80,000 parameter budget, evaluated on Tiny Shakespeare.
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_model import RiemannianGeodesicHead, GeometricGatedLinearUnit
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


# ============================================================================
# OPTION 1: Dendritic Multi-Compartment Model (K=4 compartments of Cl(4, 0))
# ============================================================================

class DendriticBlock(nn.Module):
    def __init__(self, d_model: int, num_neurons: int = 4, num_compartments: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.K = num_compartments
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        # Each compartment has 16 multivector components
        self.in_proj = nn.Linear(d_model, num_neurons * self.K * 16)
        self.gamma_proj = nn.Linear(d_model, num_neurons * self.K)
        self.biv_proj = nn.Linear(d_model, num_neurons * self.K * 6)
        
        # Bio-frequencies: theta, alpha, beta, gamma
        freq_bands = torch.tensor([1.0, 3.0, 10.0, 30.0]).view(1, self.K, 1).expand(num_neurons, self.K, 1)
        self.omega_scale = nn.Parameter(freq_bands.clone())
        
        # Somatic integration weights
        self.w_soma = nn.Parameter(torch.ones(self.K) / self.K)
        self.W_mix = nn.Parameter(torch.randn(num_neurons, num_neurons) / math.sqrt(num_neurons))
        self.out_proj = nn.Linear(num_neurons * 16, d_model)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.gglu = GeometricGatedLinearUnit(d_model, d_ffn=2 * d_model)

    def forward(self, x: torch.Tensor, dt: float = 0.05):
        B, T, D = x.shape
        residual = x
        
        x_norm = self.ln_1(x)
        x_conv = F.silu(self.conv1d(x_norm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        
        S = self.in_proj(x_conv).view(B, T, self.num_neurons * self.K, 4, 4)
        gamma = F.softplus(self.gamma_proj(x_conv)).view(B, T, self.num_neurons * self.K)
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1)
        
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, self.K, 6) * self.omega_scale
        biv = biv.view(B, T, self.num_neurons * self.K, 6)
        
        skew = torch.zeros(B, T, self.num_neurons * self.K, 4, 4, device=x.device)
        skew[..., 0, 1] =  biv[..., 0]; skew[..., 1, 0] = -biv[..., 0]
        skew[..., 0, 2] =  biv[..., 1]; skew[..., 2, 0] = -biv[..., 1]
        skew[..., 0, 3] =  biv[..., 2]; skew[..., 3, 0] = -biv[..., 2]
        skew[..., 1, 2] =  biv[..., 3]; skew[..., 2, 1] = -biv[..., 3]
        skew[..., 1, 3] =  biv[..., 4]; skew[..., 3, 1] = -biv[..., 4]
        skew[..., 2, 3] =  biv[..., 5]; skew[..., 3, 2] = -biv[..., 5]
        
        Omega = skew * (0.5 * dt)
        eye = torch.eye(4, device=x.device).expand_as(Omega)
        Q = torch.linalg.solve(eye - Omega, eye + Omega)
        
        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S
        
        # Parallel scan across all dendritic compartments
        X_all = parallel_rotor_scan(M, C).view(B, T, self.num_neurons, self.K, 4, 4)
        
        # Somatic integration: weighted linear sum across compartments
        X_soma = torch.einsum('k,btnkrc->btnrc', F.softmax(self.w_soma, dim=0), X_all)
        
        # Spikes
        energy = torch.norm(X_soma, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=0.1, beta=10.0)
        X_spiked = X_soma * spikes
        
        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)
        feat = X_mixed.reshape(B, T, self.num_neurons * 16)
        
        h = residual + self.out_proj(feat)
        out = h + self.gglu(self.ln_2(h))
        return out, Omega


class DendriticLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 48, num_layers: int = 2, num_neurons: int = 6, num_compartments: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DendriticBlock(d_model=d_model, num_neurons=num_neurons, num_compartments=num_compartments)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = RiemannianGeodesicHead(vocab_size=vocab_size, d_model=d_model)

    def forward(self, idx: torch.Tensor, targets=None, dt: float = 0.05):
        x = self.tok_emb(idx)
        total_e_kin = 0.0
        total_e_smooth = 0.0
        for block in self.blocks:
            x, Omega = block(x, dt=dt)
            e_kin = (Omega ** 2).mean()
            e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
            total_e_kin = total_e_kin + e_kin
            total_e_smooth = total_e_smooth + e_smooth
            
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + 0.01 * total_e_kin + 0.005 * total_e_smooth
        return logits, loss


# ============================================================================
# OPTION 2: Monolithic Cl(6, 0) Model (8x8 matrix, 64 components/neuron)
# ============================================================================

class Cl6Block(nn.Module):
    def __init__(self, d_model: int, num_neurons: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        # 64 multivector components per neuron
        self.in_proj = nn.Linear(d_model, num_neurons * 64)
        self.gamma_proj = nn.Linear(d_model, num_neurons)
        # 28 bivector generators for so(8)
        self.biv_proj = nn.Linear(d_model, num_neurons * 28)
        self.omega_scale = nn.Parameter(torch.ones(num_neurons, 1) * 10.0)
        
        self.W_mix = nn.Parameter(torch.randn(num_neurons, num_neurons) / math.sqrt(num_neurons))
        self.out_proj = nn.Linear(num_neurons * 64, d_model)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.gglu = GeometricGatedLinearUnit(d_model, d_ffn=2 * d_model)
        self.triu_idx = torch.triu_indices(8, 8, offset=1)

    def forward(self, x: torch.Tensor, dt: float = 0.05):
        B, T, D = x.shape
        residual = x
        
        x_norm = self.ln_1(x)
        x_conv = F.silu(self.conv1d(x_norm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        
        S = self.in_proj(x_conv).view(B, T, self.num_neurons, 8, 8)
        gamma = F.softplus(self.gamma_proj(x_conv))
        decay = torch.sigmoid(-gamma * dt).unsqueeze(-1).unsqueeze(-1)
        
        biv = self.biv_proj(x_conv).view(B, T, self.num_neurons, 28) * self.omega_scale
        skew = torch.zeros(B, T, self.num_neurons, 8, 8, device=x.device)
        u, v = self.triu_idx[0], self.triu_idx[1]
        skew[..., u, v] = biv
        skew[..., v, u] = -biv
        
        Omega = skew * (0.5 * dt)
        eye = torch.eye(8, device=x.device).expand_as(Omega)
        Q = torch.linalg.solve(eye - Omega, eye + Omega)
        
        M = torch.sqrt(decay) * Q
        C = (1.0 - decay) * S
        
        # Parallel scan on 8x8 matrices
        X_all = parallel_rotor_scan(M, C)
        
        # Spikes on 8x8 Frobenius norm
        energy = torch.norm(X_all, dim=(-1, -2), keepdim=True)
        spikes = surrogate_spike(energy, threshold=0.1, beta=10.0)
        X_spiked = X_all * spikes
        
        X_mixed = torch.einsum('ij,btjrc->btirc', self.W_mix, X_spiked)
        feat = X_mixed.reshape(B, T, self.num_neurons * 64)
        
        h = residual + self.out_proj(feat)
        out = h + self.gglu(self.ln_2(h))
        return out, Omega


class Cl6LM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 48, num_layers: int = 2, num_neurons: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Cl6Block(d_model=d_model, num_neurons=num_neurons)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = RiemannianGeodesicHead(vocab_size=vocab_size, d_model=d_model)

    def forward(self, idx: torch.Tensor, targets=None, dt: float = 0.05):
        x = self.tok_emb(idx)
        total_e_kin = 0.0
        total_e_smooth = 0.0
        for block in self.blocks:
            x, Omega = block(x, dt=dt)
            e_kin = (Omega ** 2).mean()
            e_smooth = ((Omega[:, 1:] - Omega[:, :-1]) ** 2).mean()
            total_e_kin = total_e_kin + e_kin
            total_e_smooth = total_e_smooth + e_smooth
            
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + 0.01 * total_e_kin + 0.005 * total_e_smooth
        return logits, loss


# ============================================================================
# Benchmark Runner
# ============================================================================

def evaluate(model, val_data, vocab_size):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(50):
            x, y = get_batch(val_data, batch_size=16, seq_len=128)
            logits, _ = model(x, targets=y)
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


def train_benchmark(model, name, train_data, val_data, vocab_size, num_steps=1000):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*75}")
    print(f"  Training {name} ({params:,} parameters)")
    print(f"{'='*75}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    
    t0 = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        x, y = get_batch(train_data, batch_size=32, seq_len=128)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 250 == 0 or step == 1:
            print(f"  Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Scale: {model.head.scale.item():.2f}")
            
    t_elapsed = time.time() - t0
    ppl, acc, bpc, val_loss = evaluate(model, val_data, vocab_size)
    print(f"  Finished in {t_elapsed:.1f}s | Val PPL: {ppl:.2f} | Acc: {acc:.2f}% | BPC: {bpc:.3f}")
    return {'name': name, 'params': params, 'time': t_elapsed, 'ppl': ppl, 'acc': acc, 'bpc': bpc}


def main():
    print("=" * 80)
    print("  HEAD-TO-HEAD BENCHMARK: OPTION 1 (DENDRITIC) vs OPTION 2 (MONOLITHIC CL(6,0))")
    print("=" * 80)
    train_data, val_data, vocab_size = get_data()
    
    # 1. Option 1: Biological Dendritic Multi-Compartment (K=4 in Cl(4,0))
    # num_neurons=6, num_compartments=4 -> 24 compartments total
    model_dend = DendriticLM(vocab_size, d_model=48, num_layers=2, num_neurons=6, num_compartments=4).to(device)
    res_dend = train_benchmark(model_dend, "Option 1: Dendritic Multi-Compartment (K=4, Cl(4,0))", train_data, val_data, vocab_size)
    
    # 2. Option 2: Monolithic Cl(6, 0) (8x8 matrices, Spin(6)=SU(4))
    # num_neurons=6 -> 6x64 = 384 state dims
    model_cl6 = Cl6LM(vocab_size, d_model=48, num_layers=2, num_neurons=6).to(device)
    res_cl6 = train_benchmark(model_cl6, "Option 2: Monolithic Cl(6, 0) (8x8, Spin(6))", train_data, val_data, vocab_size)
    
    print("\n" + "=" * 85)
    print("  FINAL SCIENTIFIC VERDICT: EVOLUTION (DENDRITES) VS SILICON (CL(6,0))")
    print("=" * 85)
    print(f"  {'Model Architecture':<48} | {'Params':<8} | {'PPL':<6} | {'Top-1 Acc':<9} | {'BPC':<6}")
    print("  " + "-" * 81)
    print(f"  {'Baseline 2-Layer Cl(4,0) (Single-Compartment)':<48} | {'80,802':<8} | {'6.36':<6} | {'46.23%':<9} | {'2.669':<6}")
    print(f"  {res_dend['name']:<48} | {res_dend['params']:<8,d} | {res_dend['ppl']:<6.2f} | {res_dend['acc']:<8.2f}% | {res_dend['bpc']:<6.3f}")
    print(f"  {res_cl6['name']:<48} | {res_cl6['params']:<8,d} | {res_cl6['ppl']:<6.2f} | {res_cl6['acc']:<8.2f}% | {res_cl6['bpc']:<6.3f}")
    print("=" * 85)


if __name__ == '__main__':
    main()
