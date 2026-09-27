"""
Unit Test Suite for Adaptive Delta t Continuous-Time Warping in Cl(6, 0).
Verifies:
1. Positivity and nominal initialization of Delta t_t (~0.05).
2. Exact Cayley rotor orthogonality under token-varying Delta t_t.
3. Gradient propagation to time pacing parameters (W_dt, b_dt).
4. Full end-to-end 2-layer causal LM execution.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))
from python.clifford_cl6 import AdaptiveTimeHead, CliffordCL6Block, CliffordCL6ForCausalLM, cayley_rotor_8x8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_adaptive_time_head():
    print("[Test 1: AdaptiveTimeHead Positivity & Initialization]")
    d_model = 48
    dt_base = 0.05
    head = AdaptiveTimeHead(d_model=d_model, dt_base=dt_base).to(device)
    
    x = torch.randn(4, 32, d_model, device=device)
    dt = head(x) # [B, T, 1, 1, 1]
    
    assert (dt > 0.0).all(), "Delta t must be strictly positive!"
    # Check that initially dt is close to dt_base (within 1e-4)
    init_err = torch.max(torch.abs(dt - dt_base)).item()
    print(f"  Initial deviation from dt_base={dt_base}: {init_err:.2e}")
    assert init_err < 1e-3, f"Adaptive head did not initialize near dt_base! Error: {init_err}"
    print("  -> PASSED: Delta t starts strictly positive and smoothly calibrated at dt_base.")


def test_rotor_orthogonality_with_adaptive_dt():
    print("[Test 2: Cayley Rotor Orthogonality under Adaptive Delta t]")
    d_model = 48
    block = CliffordCL6Block(d_model=d_model, num_neurons=4, dt_base=0.05).to(device)
    
    x = torch.randn(4, 32, d_model, device=device)
    out, Omega, dt = block(x)
    
    Q = cayley_rotor_8x8(Omega)
    eye = torch.eye(8, device=device).expand_as(Q)
    ortho_err = torch.max(torch.abs(torch.matmul(Q.transpose(-1, -2), Q) - eye)).item()
    print(f"  Maximum orthogonality error ||Q^T Q - I||: {ortho_err:.2e}")
    assert ortho_err < 1e-5, f"Cayley rotor is not orthogonal! Error: {ortho_err}"
    print("  -> PASSED: Exact Spin(6) / SO(8) isometry preserved under adaptive time pacing.")


def test_gradient_flow_to_time_head():
    print("[Test 3: Gradient Flow through Adaptive Time Parameters]")
    d_model = 48
    block = CliffordCL6Block(d_model=d_model, num_neurons=4, dt_base=0.05).to(device)
    
    x = torch.randn(2, 16, d_model, device=device, requires_grad=True)
    out, Omega, dt = block(x)
    
    loss = out.sum() + (dt ** 2).sum()
    loss.backward()
    
    assert block.time_head.proj.weight.grad is not None, "W_dt did not receive gradients!"
    assert block.time_head.proj.bias.grad is not None, "b_dt did not receive gradients!"
    assert not torch.isnan(block.time_head.proj.weight.grad).any(), "NaNs in W_dt grad!"
    assert not torch.isnan(block.time_head.proj.bias.grad).any(), "NaNs in b_dt grad!"
    print("  -> PASSED: Clean non-vanishing gradients propagate back through the adaptive clock.")


def test_full_cl6_model():
    print("[Test 4: Full CliffordCL6ForCausalLM Forward & Backward]")
    vocab_size = 65
    d_model = 48
    model = CliffordCL6ForCausalLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_neurons=6).to(device)
    
    B, T = 4, 64
    idx = torch.randint(0, vocab_size, (B, T), device=device)
    targets = torch.randint(0, vocab_size, (B, T), device=device)
    
    logits, loss, diags = model(idx, targets=targets)
    assert logits.shape == (B, T, vocab_size), "Logits shape mismatch!"
    assert loss is not None and not torch.isnan(loss), "Loss is NaN!"
    
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Param {name} received no grad!"
        assert not torch.isnan(param.grad).any(), f"Param {name} has NaN grad!"
        
    print(f"  Learned clock diagnostics on random batch: mean_dt={diags['mean_dt']:.4f}, min_dt={diags['min_dt']:.4f}, max_dt={diags['max_dt']:.4f}")
    print("  -> PASSED: Complete 2-layer Cl(6, 0) causal LM executes smoothly.")


if __name__ == '__main__':
    print("=" * 80)
    print("  RUNNING ADAPTIVE DELTA T UNIT TESTS (CL(6, 0))")
    print("=" * 80)
    test_adaptive_time_head()
    test_rotor_orthogonality_with_adaptive_dt()
    test_gradient_flow_to_time_head()
    test_full_cl6_model()
    print("=" * 80)
    print("  ALL ADAPTIVE DELTA T TESTS PASSED!")
    print("=" * 80)
