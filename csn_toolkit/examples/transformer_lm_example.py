"""
Example: Causal Language Modeling with Clifford Transformer (CSN-LM).

Demonstrates:
1. Instantiating CliffordTransformerLM (replacing Attention with Spin(4) Associative Scans)
2. Causal autoregressive token prediction
3. Cross-entropy loss calculation & backpropagation
4. Autoregressive greedy text generation
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

toolkit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(toolkit_path))

from csn import CliffordTransformerLM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")

    vocab_size = 4096
    d_model = 128
    num_layers = 4
    num_heads = 4

    # 1. Initialize Causal Clifford Language Model
    model = CliffordTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        rotor_dim=4,           # Spin(4) Lie Rotors
        max_seq_len=1024
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Initialized Clifford LM with {total_params:,} parameters")

    # 2. Synthetic Batch of Token Sequences (Batch: 4, Length: 64)
    B, L = 4, 64
    input_ids = torch.randint(0, vocab_size, (B, L), device=device)
    targets = torch.randint(0, vocab_size, (B, L), device=device)

    # 3. Forward Pass & Causal Language Modeling Loss
    logits = model(input_ids)  # [B, L, vocab_size]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    print(f"[INFO] Forward pass complete! CrossEntropy Loss: {loss.item():.4f}")

    # 4. Backward Pass & Optimizer Update
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[INFO] Backward pass & optimizer step complete!")

    # 5. Autoregressive Text Generation (Greedy Decoding)
    prompt = torch.tensor([[10, 25, 42]], device=device)
    generated = prompt.clone()
    print(f"\n[INFO] Autoregressive Generation from seed {prompt.tolist()[0]}:")
    model.eval()
    with torch.no_grad():
        for step in range(10):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    print(f"Generated Token Sequence: {generated.tolist()[0]}")
    print("[SUCCESS] Clifford Transformer LM is fully operational!")

if __name__ == "__main__":
    main()
