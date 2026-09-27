"""
Empirical Benchmark & Linguistic Analysis of Adaptive Continuous-Time Warping (Delta t(t))
in Monolithic Cl(6, 0) Clifford-TENN Architecture.

Evaluates:
1. Convergence, validation PPL, Top-1 Accuracy, and BPC.
2. Direct comparison with Fixed Delta t = 0.05 baseline.
3. Linguistic Clock Analysis: Which characters/tokens cause the model to decelerate or accelerate?
"""

import os
import sys
import math
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_cl6 import CliffordCL6ForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data():
    file_path = os.path.join('data', 'tinyshakespeare.txt')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, char_to_idx, idx_to_char


def get_batch(data, batch_size=32, seq_len=128):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


def evaluate(model, val_data, vocab_size, idx_to_char):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    
    char_dt_values = defaultdict(list)
    
    with torch.no_grad():
        for _ in range(50):
            x, y = get_batch(val_data, batch_size=16, seq_len=128)
            logits, loss, diags = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()
            
            # Extract layer 0 dt predictions per character
            x_conv = model.blocks[0].conv1d(model.blocks[0].ln_1(model.tok_emb(x)).transpose(1, 2))[:, :, :128].transpose(1, 2)
            dt_vals = model.blocks[0].time_head(F.silu(x_conv)).squeeze().cpu() # [B, T]
            x_cpu = x.cpu()
            for b in range(x.shape[0]):
                for t in range(x.shape[1]):
                    ch = idx_to_char[x_cpu[b, t].item()]
                    char_dt_values[ch].append(dt_vals[b, t].item())

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    acc = 100.0 * total_correct / total_tokens
    bpc = avg_loss / math.log(2)
    return ppl, acc, bpc, avg_loss, char_dt_values


def train_benchmark(num_steps=1000):
    print("=" * 85)
    print(f"  BENCHMARK: MONOLITHIC CL(6, 0) WITH ADAPTIVE DELTA T(t) ON {device.type.upper()}")
    print("=" * 85)
    
    train_data, val_data, vocab_size, char_to_idx, idx_to_char = get_data()
    
    # 2-layer Cl(6, 0) model with Adaptive Time
    model = CliffordCL6ForCausalLM(
        vocab_size=vocab_size,
        d_model=48,
        num_layers=2,
        num_neurons=6,
        dt_base=0.05,
        lambda_kin=0.01,
        lambda_smooth=0.005,
        lambda_time=0.01
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    
    t0 = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        x, y = get_batch(train_data, batch_size=32, seq_len=128)
        optimizer.zero_grad()
        
        logits, loss, diags = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 250 == 0 or step == 1:
            print(f"  Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | CE: {diags['ce_loss']:.4f} | Act: {diags['action_loss']:.4f} | dt: [{diags['min_dt']:.4f}, {diags['mean_dt']:.4f}, {diags['max_dt']:.4f}]")
            
    train_time = time.time() - t0
    print(f"  Training finished in {train_time:.1f}s.")
    
    ppl, acc, bpc, val_loss, char_dt_values = evaluate(model, val_data, vocab_size, idx_to_char)
    
    print("\n" + "=" * 85)
    print("  EMPIRICAL RESULTS: ADAPTIVE DELTA T(t) vs FIXED DELTA T = 0.05")
    print("=" * 85)
    print(f"  {'Model Architecture':<45} | {'Params':<8} | {'PPL':<6} | {'Top-1 Acc':<9} | {'BPC':<6}")
    print("  " + "-" * 81)
    print(f"  {'Fixed Delta t = 0.05 (Cl(6, 0))':<45} | {'126,642':<8} | {'5.76':<6} | {'48.41%':<9} | {'2.526':<6}")
    print(f"  {'Adaptive Delta t(t) (Cl(6, 0))':<45} | {num_params:<8,d} | {ppl:<6.2f} | {acc:<8.2f}% | {bpc:<6.3f}")
    print("=" * 85)
    
    # Character Clock Diagnostics
    print("\n" + "=" * 85)
    print("  LINGUISTIC CONTINUOUS-TIME PACING ANALYSIS (Learned Delta t per Token)")
    print("=" * 85)
    
    avg_dts = {ch: (sum(v) / len(v), len(v)) for ch, v in char_dt_values.items() if len(v) > 20}
    sorted_by_speed = sorted(avg_dts.items(), key=lambda x: x[1][0])
    
    print("\n  [Top 8 'Slow Time' Tokens - Model Decelerates Internal Clock For Deep Processing]:")
    print(f"  {'Char':<6} | {'Display':<12} | {'Mean Delta t':<14} | {'Relative to 0.05':<16} | {'Count'}")
    print("  " + "-" * 65)
    for ch, (mean_dt, count) in sorted_by_speed[:8]:
        disp = repr(ch)
        rel = (mean_dt / 0.05 - 1.0) * 100.0
        print(f"  {ch:<6} | {disp:<12} | {mean_dt:<14.5f} | {rel:>+7.2f}%         | {count}")

    print("\n  [Top 8 'Fast Time' Tokens - Model Accelerates Clock Across Predictable Sequences]:")
    print(f"  {'Char':<6} | {'Display':<12} | {'Mean Delta t':<14} | {'Relative to 0.05':<16} | {'Count'}")
    print("  " + "-" * 65)
    for ch, (mean_dt, count) in sorted_by_speed[-8:]:
        disp = repr(ch)
        rel = (mean_dt / 0.05 - 1.0) * 100.0
        print(f"  {ch:<6} | {disp:<12} | {mean_dt:<14.5f} | {rel:>+7.2f}%         | {count}")
    print("=" * 85)


if __name__ == '__main__':
    train_benchmark()
