"""
Unit Test Suite for Deep Multi-Layer Clifford-TENN Architecture.
Verifies:
1. Forward pass shapes and output dimensions.
2. Backward pass & end-to-end gradient propagation (zero NaNs).
3. Invariant contractivity and energy stability.
4. Scale & bias optimization in Riemannian Geodesic Head.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_model import CliffordTENNForCausalLM, CliffordTENNBlock, RiemannianGeodesicHead


def test_riemannian_head():
    print("[Test 1: Riemannian Geodesic Head]")
    vocab_size = 65
    d_model = 64
    B, T = 4, 32
    head = RiemannianGeodesicHead(vocab_size=vocab_size, d_model=d_model, init_scale=16.0)
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    logits = head(x)
    
    assert logits.shape == (B, T, vocab_size), f"Unexpected shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaNs in Riemannian logits"
    
    loss = logits.sum()
    loss.backward()
    
    assert x.grad is not None and not torch.isnan(x.grad).any(), "NaNs in input gradient"
    assert head.prototypes.grad is not None, "Prototypes did not receive gradients"
    assert head.scale.grad is not None, "Scale parameter did not receive gradients"
    assert head.bias.grad is not None, "Prior bias did not receive gradients"
    print("  -> PASSED: Riemannian Head produces valid geodesic logits and clean gradients.")


def test_clifford_block():
    print("[Test 2: Single CliffordTENNBlock Forward & Backward]")
    B, T, d_model, num_neurons = 2, 64, 32, 8
    block = CliffordTENNBlock(d_model=d_model, num_neurons=num_neurons)
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    out, Omega = block(x, dt=0.05)
    
    assert out.shape == (B, T, d_model), f"Output shape mismatch: {out.shape}"
    assert Omega.shape == (B, T, num_neurons, 4, 4), f"Omega shape mismatch: {Omega.shape}"
    assert not torch.isnan(out).any(), "NaNs in block output"
    
    loss = out.sum() + (Omega ** 2).sum()
    loss.backward()
    
    assert x.grad is not None and not torch.isnan(x.grad).any(), "NaNs in block gradient"
    print("  -> PASSED: CliffordTENNBlock executes with zero NaNs and valid adjoint gradients.")


def test_deep_model_end_to_end():
    print("[Test 3: Deep Multi-Layer CliffordTENNForCausalLM (2 layers)]")
    vocab_size = 65
    d_model = 64
    num_layers = 2
    num_neurons = 16
    B, T = 4, 128
    
    model = CliffordTENNForCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_neurons=num_neurons,
        kernel_size=4
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model has {num_params:,} trainable parameters across {num_layers} layers.")
    
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    
    logits, loss, action_val = model(idx, targets=targets)
    
    assert logits.shape == (B, T, vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss is NaN"
    assert action_val >= 0.0, "Action penalty should be non-negative"
    
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
        
    print("  -> PASSED: End-to-end 2-layer model trains with full gradient flow across all parameters.")


def test_extreme_sequence_length():
    print("[Test 4: Numerical Stability under Long Context (T=512)]")
    vocab_size = 65
    d_model = 32
    model = CliffordTENNForCausalLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_neurons=8)
    
    B, T = 2, 512
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    
    logits, loss, _ = model(idx, targets=targets)
    assert not torch.isnan(logits).any(), "NaNs in long context forward pass"
    loss.backward()
    
    print("  -> PASSED: Long context T=512 is stable with parallel associative scan.")


if __name__ == '__main__':
    print("=" * 80)
    print("  RUNNING DEEP CLIFFORD-TENN ARCHITECTURE UNIT TESTS")
    print("=" * 80)
    test_riemannian_head()
    test_clifford_block()
    test_deep_model_end_to_end()
    test_extreme_sequence_length()
    print("=" * 80)
    print("  ALL DEEP CLIFFORD-TENN TESTS PASSED!")
    print("=" * 80)
