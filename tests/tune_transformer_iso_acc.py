"""
Transformer Iso-Accuracy Target Trainer:
Trains Scaled Causal Transformer (NanoGPT style, 1.90M params)
until it reaches the exact target accuracy of Clifford-TENN (~50% Top-1 Accuracy, PPL ~5.5).

Evaluates validation accuracy every 200 steps and records the exact training steps,
time, and throughput required to reach parity with Clifford-TENN Cl(8, 0).
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # GPT-2 standard residual initialization scaling
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))

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


class ScaledTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 192, n_heads: int = 6, n_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': CausalSelfAttention(d_model, n_heads, n_layers),
                'ln2': nn.LayerNorm(d_model),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model)
                )
            }) for _ in range(n_layers)
        ])
        # Residual scaling for MLP
        for layer in self.layers:
            nn.init.normal_(layer['mlp'][2].weight, std=0.02 / math.sqrt(2 * n_layers))

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        h = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            h = h + layer['attn'](layer['ln1'](h))
            h = h + layer['mlp'](layer['ln2'](h))
        logits = self.head(self.ln_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def evaluate(model, val_data):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(30):
            x, y = get_batch(val_data, batch_size=16, seq_len=128)
            logits, loss = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc = 100.0 * total_correct / total_tokens
    bpc = avg_loss / math.log(2)
    return ppl, acc, bpc, avg_loss


def profile_speed_and_memory(model):
    model.eval()
    B, T = 1, 128
    dummy_input = torch.randint(0, 65, (B, T), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Latency & Throughput
    iters = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    latency_ms = (total_time / iters) * 1000.0
    throughput = (iters * T) / total_time
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return throughput, latency_ms, peak_vram_mb


def train():
    train_data, val_data, vocab_size = get_data()

    # Scaled Transformer (1.90M parameters)
    model = ScaledTransformerLM(
        vocab_size=vocab_size,
        d_model=192,
        n_heads=6,
        n_layers=4,
        max_seq_len=512
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 80)
    print(f"  TARGET TRAINING: SCALED TRANSFORMER ({num_params:,} parameters)")
    print(f"  Target: Match Clifford-TENN accuracy (>= 49.5% - 50.5%)")
    print("=" * 80)

    max_steps = 2500
    warmup_steps = 100
    base_lr = 2.0e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.05)

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 1e-4 + 0.5 * (base_lr - 1e-4) * (1.0 + math.cos(math.pi * progress))

    t0 = time.time()
    reached_target = False
    best_acc = 0.0

    for step in range(1, max_steps + 1):
        model.train()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(train_data, batch_size=32, seq_len=128)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == 1:
            ppl, acc, bpc, val_loss = evaluate(model, val_data)
            elapsed = time.time() - t0
            print(f"  Step {step:4d}/{max_steps} ({elapsed:5.1f}s) | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {ppl:5.2f} | Acc: {acc:5.2f}%")
            if acc > best_acc:
                best_acc = acc

            # Target threshold check: matching Clifford-TENN (>= 49.5%)
            if acc >= 49.5 and not reached_target:
                reached_target = True
                print("\n  " + "*" * 60)
                print(f"  >>> TARGET REACHED AT STEP {step}! Val Acc: {acc:.2f}%, PPL: {ppl:.2f}")
                print("  " + "*" * 60 + "\n")
                if acc >= 50.5:
                    break

    train_time = time.time() - t0
    final_ppl, final_acc, final_bpc, final_val_loss = evaluate(model, val_data)
    throughput, latency_ms, peak_vram_mb = profile_speed_and_memory(model)

    print("\n" + "=" * 80)
    print("  FINAL TRANSFORMER ISO-ACCURACY RESULTS")
    print("=" * 80)
    print(f"  Parameters:     {num_params:,}")
    print(f"  Training Time:  {train_time:.1f}s")
    print(f"  Val Loss:       {final_val_loss:.4f}")
    print(f"  Val PPL:        {final_ppl:.2f}")
    print(f"  Top-1 Accuracy: {final_acc:.2f}%")
    print(f"  Bits/Char:      {final_bpc:.3f}")
    print(f"  Throughput:     {throughput:.0f} tok/s")
    print(f"  Latency:        {latency_ms:.2f} ms/token")
    print(f"  Peak VRAM:      {peak_vram_mb:.1f} MB")
    print("=" * 80)


if __name__ == '__main__':
    train()
