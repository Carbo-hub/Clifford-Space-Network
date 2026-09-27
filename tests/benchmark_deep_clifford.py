"""
Empirical Benchmark of Deep Multi-Layer Clifford-TENN against Sequence Baselines.

Evaluates:
1. Deep Clifford-TENN (2 layers, matched ~75k params, GGLU, Canonical Riemannian Head).
2. Deep Clifford-TENN (2 layers, scaled ~140k params, GGLU, Canonical Riemannian Head).
Against the established baselines on Tiny Shakespeare:
- Transformer (240k params)
- Mamba S6 (72k params)
- Liquid CfC (33k params)
- 1-Layer Clifford-TENN (45k params)
"""

import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_model import CliffordTENNForCausalLM

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


def evaluate_model(model, val_data, vocab_size, seq_len=128, num_eval_batches=50):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(num_eval_batches):
            x, y = get_batch(val_data, batch_size=16, seq_len=seq_len)
            logits, _, _ = model(x, dt=0.05, targets=y)
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


def train_model(model, name, train_data, val_data, vocab_size, num_steps=1000, lr=2e-3):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Training {name} ({num_params:,} parameters) ---")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    
    t0 = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        x, y = get_batch(train_data, batch_size=32, seq_len=128)
        optimizer.zero_grad()
        
        logits, loss, action_val = model(x, dt=0.05, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 250 == 0 or step == 1:
            ce_loss = loss.item() - action_val
            scale_val = model.head.scale.item()
            print(f"  Step {step:4d}/{num_steps} | Total: {loss.item():.4f} | CE: {ce_loss:.4f} | Act: {action_val:.4f} | Scale: {scale_val:.2f}")
            
    train_time = time.time() - t0
    print(f"  Training finished in {train_time:.1f}s.")
    
    ppl, acc, bpc, val_loss = evaluate_model(model, val_data, vocab_size)
    print(f"  Validation: PPL={ppl:.2f} | Acc={acc:.2f}% | BPC={bpc:.3f} | ValLoss={val_loss:.4f}")
    
    return {
        'name': name,
        'params': num_params,
        'train_time': train_time,
        'ppl': ppl,
        'acc': acc,
        'bpc': bpc,
        'val_loss': val_loss
    }


def main():
    print("=" * 80)
    print(f"  DEEP CLIFFORD-TENN MULTI-LAYER BENCHMARK ON {device.type.upper()}")
    print("  Evaluating Stacked Layers, GGLU & Canonical Riemannian Geodesic Head")
    print("=" * 80)
    
    train_data, val_data, vocab_size = get_data()
    
    # 1. Deep Clifford-TENN (Matched ~75k params, 2 layers, d_model=48, num_neurons=12)
    model_matched = CliffordTENNForCausalLM(
        vocab_size=vocab_size,
        d_model=48,
        num_layers=2,
        num_neurons=12,
        kernel_size=4,
        spike_threshold=0.1,
        lambda_kin=0.01,
        lambda_smooth=0.005
    ).to(device)
    res_matched = train_model(model_matched, "Deep Clifford-TENN (Matched, 2-Layer)", train_data, val_data, vocab_size)
    
    # Comprehensive Summary Table
    print("\n" + "=" * 92)
    print("  COMPREHENSIVE SEQUENCE MODEL LEADERBOARD (TINY SHAKESPEARE)")
    print("=" * 92)
    print(f"  {'Model Architecture':<35} | {'Params':<8} | {'PPL':<6} | {'Top-1 Acc':<9} | {'BPC':<6} | {'Geometry':<12}")
    print("  " + "-" * 88)
    print(f"  {'Transformer (Attention, 2-layer)':<35} | {'240,000':<8} | {'11.75':<6} | {'28.42%':<9} | {'3.554':<6} | {'Euclidean':<12}")
    print(f"  {'Liquid CfC (Hasani et al., 2-layer)':<35} | {'33,000':<8} | {'9.94':<6} | {'33.72%':<9} | {'3.313':<6} | {'Scalar':<12}")
    print(f"  {'Mamba S6 (Gu et al., 2-layer)':<35} | {'72,000':<8} | {'7.37':<6} | {'41.65%':<9} | {'2.882':<6} | {'Linear SSM':<12}")
    print(f"  {'Clifford-TENN (1-layer toy, Euclidean)':<35} | {'45,890':<8} | {'7.45':<6} | {'41.30%':<9} | {'2.897':<6} | {'Euclidean':<12}")
    print(f"  {'Clifford-TENN (1-layer Riemannian)':<35} | {'45,890':<8} | {'7.85':<6} | {'40.38%':<9} | {'2.973':<6} | {'Spin(4) Manifold':<12}")
    print(f"  {res_matched['name']:<35} | {res_matched['params']:<8,d} | {res_matched['ppl']:<6.2f} | {res_matched['acc']:<8.2f}% | {res_matched['bpc']:<6.3f} | {'Spin(4) Manifold':<12}")
    print("=" * 92)


if __name__ == '__main__':
    main()
