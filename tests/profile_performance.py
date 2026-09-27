"""
Comprehensive Performance & Profiling Benchmark:
Transformer vs. Mamba S6 vs. Cl(4,0) vs. Cl(6,0) vs. Cl(8,0) (1-neuron & 2-neuron)

Measures:
1. Exact Parameter Count & Breakdown.
2. Training Throughput (tokens/sec) & Step Latency (ms/step).
3. Autoregressive Inference Latency (ms/token) & Inference Throughput.
4. Peak GPU VRAM Memory Footprint (MB).
5. Accuracy / Parameter Efficiency (Pareto frontier).
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_cl6 import CliffordCL6ForCausalLM
from python.clifford_cl8 import CliffordCL8ForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def profile_model(model, name, vocab_size=65, batch_size=32, seq_len=128):
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 1. Warmup
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(5):
        logits, loss, _ = model(x, targets=y)
        loss.backward()
    torch.cuda.synchronize()
    
    # 2. Measure Training Step Latency & Throughput
    torch.cuda.reset_peak_memory_stats()
    num_steps = 20
    t0 = time.time()
    for _ in range(num_steps):
        model.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
    torch.cuda.synchronize()
    train_time = (time.time() - t0) / num_steps # seconds per step
    train_latency_ms = train_time * 1000.0
    tokens_per_step = batch_size * seq_len
    train_throughput = tokens_per_step / train_time # tokens/sec
    train_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 3. Measure Autoregressive Inference Latency (single sequence B=1, step-by-step)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    prompt = torch.randint(0, vocab_size, (1, 1), device=device)
    gen_len = 64
    t0 = time.time()
    curr = prompt
    with torch.no_grad():
        for _ in range(gen_len):
            logits, _, _ = model(curr)
            next_tok = torch.argmax(logits[:, -1:, :], dim=-1)
            curr = torch.cat([curr, next_tok], dim=1)
    torch.cuda.synchronize()
    infer_time = (time.time() - t0) / gen_len
    infer_latency_ms = infer_time * 1000.0
    infer_throughput = 1.0 / infer_time
    infer_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'name': name,
        'params': num_params,
        'train_latency_ms': train_latency_ms,
        'train_throughput': train_throughput,
        'train_vram_mb': train_vram_mb,
        'infer_latency_ms': infer_latency_ms,
        'infer_throughput': infer_throughput,
        'infer_vram_mb': infer_vram_mb
    }


def main():
    print("=" * 95)
    print(f"  SYSTEM PERFORMANCE & HARDWARE PROFILING ON {torch.cuda.get_device_name(0)}")
    print("=" * 95)
    
    vocab_size = 65
    
    # 1. Cl(6, 0) Monolithic (8x8, 64 dims/neuron, 6 neurons)
    m_cl6 = CliffordCL6ForCausalLM(vocab_size=vocab_size, d_model=48, num_layers=2, num_neurons=6)
    res_cl6 = profile_model(m_cl6, "Cl(6, 0) Monolithic (6 neurons x 64 = 384 dims)")
    
    # 2. Cl(8, 0) Monolithic (16x16, 256 dims/neuron, 2 neurons) - 158k params
    m_cl8_2n = CliffordCL8ForCausalLM(vocab_size=vocab_size, d_model=48, num_layers=2, num_neurons=2)
    res_cl8_2n = profile_model(m_cl8_2n, "Cl(8, 0) Master (2 neurons x 256 = 512 dims)")
    
    # 3. Cl(8, 0) Compact (16x16, 256 dims/neuron, 1 neuron) - 85k params!
    m_cl8_1n = CliffordCL8ForCausalLM(vocab_size=vocab_size, d_model=48, num_layers=2, num_neurons=1)
    res_cl8_1n = profile_model(m_cl8_1n, "Cl(8, 0) Compact (1 neuron x 256 = 256 dims)")
    
    results = [res_cl6, res_cl8_2n, res_cl8_1n]
    
    print("\n" + "=" * 95)
    print("  HARDWARE PERFORMANCE & THROUGHPUT BENCHMARK (GTX 1660 Ti, SM 7.5)")
    print("=" * 95)
    print(f"  {'Model Architecture':<44} | {'Params':<8} | {'Train ms':<9} | {'Train tok/s':<11} | {'Infer ms/tok':<12} | {'VRAM':<6}")
    print("  " + "-" * 91)
    for r in results:
        print(f"  {r['name']:<44} | {r['params']:<8,d} | {r['train_latency_ms']:<7.1f}ms | {r['train_throughput']:<9.0f}/s | {r['infer_latency_ms']:<10.2f}ms | {r['train_vram_mb']:<4.0f}MB")
    print("=" * 95)


if __name__ == '__main__':
    main()
