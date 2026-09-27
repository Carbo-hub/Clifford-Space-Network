"""
Example: Object Detection with Clifford Space Network (CSN-Large).

Demonstrates:
1. Model initialization
2. Multi-scale forward pass (P3, P4, P5)
3. Sasaki Contact Metric Loss calculation
4. Backward pass & ModelEMA update
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim

# Add toolkit to path
toolkit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(toolkit_path))

from csn import CliffordDetModel, ModelEMA, compute_sasaki_ciou_vectorized

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")

    # 1. Initialize CSN-Large Vision Model
    model = CliffordDetModel(
        num_classes=1,       # E.g. Pedestrian / Single-class or COCO 80 classes
        base_channels=64,    # Large: 64, Medium: 48, Nano: 32
        num_neurons_cl4=2,   # Spin(4) neurons in C3
        num_neurons_cl8=1,   # Spin(8) neurons in C4 & C5
        use_schulz=True      # Asynchronous non-blocking Schulz inversion
    ).to(device)

    ema = ModelEMA(model, decay=0.999)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model initialized with {total_params:,} parameters ({total_params*4/(1024*1024):.2f} MB)")

    # 2. Synthetic Batch (Batch Size: 2, 3x640x640)
    B, C, H, W = 2, 3, 640, 640
    images = torch.randn(B, C, H, W, device=device)

    print(f"[INFO] Executing Forward Pass for input {images.shape}...")
    preds = model(images)

    for scale, out in preds.items():
        hm = out["heatmap"]
        wh = out["wh"]
        off = out["offset"]
        print(f"  Scale {scale.upper()}: Heatmap={tuple(hm.shape)}, WH={tuple(wh.shape)}, Offset={tuple(off.shape)}")

    # 3. Compute Example Sasaki Geodesic Contact Loss
    # Synthetic predicted and ground truth boxes: (x1, y1, x2, y2)
    pred_boxes = torch.tensor([
        [100.0, 150.0, 200.0, 350.0],
        [300.0, 200.0, 420.0, 480.0]
    ], device=device)

    gt_boxes = torch.tensor([
        [105.0, 148.0, 205.0, 352.0],
        [295.0, 205.0, 415.0, 475.0]
    ], device=device)

    d_sasaki = compute_sasaki_ciou_vectorized(pred_boxes, gt_boxes, kappa_reeb=0.85)
    loss_sasaki = d_sasaki.mean()
    print(f"[INFO] Sasaki Contact Metric Distance: {loss_sasaki.item():.4f}")

    # 4. Dummy Multi-Scale Center Loss
    loss_hm = sum(out["heatmap"].mean() for out in preds.values())
    total_loss = loss_hm + 2.0 * loss_sasaki

    # 5. Backward Pass & Step
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    ema.update(model)

    print(f"[SUCCESS] Step complete! Loss = {total_loss.item():.4f}")

if __name__ == "__main__":
    main()
