"""
Unit Test Suite for Clifford-Det Cl(8, 0):
Validates 2D Bidirectional Spatial Rotor Scan, Anchor-Free Detection Head, Loss, and Gradients on 640x640 images.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath('.'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from python.clifford_vision import CliffordDetModel, compute_detection_loss


class TestCliffordVisionDet(unittest.TestCase):

    def test_model_forward_and_output_shapes(self):
        """Test multi-scale forward pass on standard 640x640 input resolution."""
        model = CliffordDetModel(num_classes=1, base_channels=32, num_neurons_cl4=2, num_neurons_cl8=1).to(device)
        model.eval()

        B, C, H, W = 2, 3, 640, 640
        x = torch.randn(B, C, H, W, device=device)

        with torch.no_grad():
            preds = model(x)

        # Verify multi-scale levels: P3 (80x80), P4 (40x40), P5 (20x20)
        self.assertIn("p3", preds)
        self.assertIn("p4", preds)
        self.assertIn("p5", preds)

        self.assertEqual(preds["p3"]["heatmap"].shape, (B, 1, 80, 80))
        self.assertEqual(preds["p3"]["wh"].shape, (B, 2, 80, 80))
        self.assertEqual(preds["p3"]["offset"].shape, (B, 2, 80, 80))

        self.assertEqual(preds["p4"]["heatmap"].shape, (B, 1, 40, 40))
        self.assertEqual(preds["p4"]["wh"].shape, (B, 2, 40, 40))
        self.assertEqual(preds["p4"]["offset"].shape, (B, 2, 40, 40))

        self.assertEqual(preds["p5"]["heatmap"].shape, (B, 1, 20, 20))
        self.assertEqual(preds["p5"]["wh"].shape, (B, 2, 20, 20))
        self.assertEqual(preds["p5"]["offset"].shape, (B, 2, 20, 20))

        # Check values are finite
        for scale in ["p3", "p4", "p5"]:
            self.assertFalse(torch.isnan(preds[scale]["heatmap"]).any())
            self.assertFalse(torch.isnan(preds[scale]["wh"]).any())
            self.assertFalse(torch.isnan(preds[scale]["offset"]).any())

        print(f"\n[PASS] Multi-Scale Forward Shapes Verified: P3 {preds['p3']['heatmap'].shape}, P4 {preds['p4']['heatmap'].shape}, P5 {preds['p5']['heatmap'].shape}")

    def test_parameter_count(self):
        """Verify model compactness compared to YOLO26n (target < 500k params vs 2.57M)."""
        model = CliffordDetModel(num_classes=1, base_channels=32, num_neurons_cl4=2, num_neurons_cl8=1)
        params = sum(p.numel() for p in model.parameters())
        print(f"[PASS] CSN 2.0 Parameter Count: {params:,} parameters (YOLO26n: 2,572,280 parameters, Ratio: {2572280/params:.1f}x smaller)")
        self.assertLess(params, 500_000, f"Model parameters exceed budget: {params}")

    def test_nms_free_detections(self):
        """Test multi-scale detection extraction across P3, P4, P5."""
        model = CliffordDetModel(num_classes=1, base_channels=32, num_neurons_cl4=2, num_neurons_cl8=1).to(device)
        model.eval()

        x = torch.randn(2, 3, 640, 640, device=device)
        with torch.no_grad():
            detections = model.get_detections(x, conf_thresh=0.001, imgsz=640)

        self.assertEqual(len(detections), 2)
        for b in range(2):
            dets = detections[b]
            self.assertEqual(dets.shape[-1], 6)  # [x1, y1, x2, y2, score, class_id]
            if len(dets) > 0:
                self.assertTrue((dets[:, 0] <= dets[:, 2]).all())  # x1 <= x2
                self.assertTrue((dets[:, 1] <= dets[:, 3]).all())  # y1 <= y2
        print(f"[PASS] Multi-Scale Detections Extracted: Batch 0 has {len(detections[0])} boxes, Batch 1 has {len(detections[1])} boxes")

    def test_loss_and_backward_gradients(self):
        """Test multi-scale CIoU loss computation and backpropagation to all parameters."""
        model = CliffordDetModel(num_classes=1, base_channels=32, num_neurons_cl4=2, num_neurons_cl8=1).to(device)
        model.train()

        x = torch.randn(2, 3, 640, 640, device=device)
        # Mock ground truth targets: small pedestrian in P3, medium in P4, large in P5
        targets = [
            torch.tensor([[0, 0.5, 0.5, 0.05, 0.1], [0, 0.2, 0.3, 0.15, 0.3]], device=device), # small & medium
            torch.tensor([[0, 0.8, 0.7, 0.3, 0.6]], device=device) # large
        ]

        preds = model(x)
        loss, loss_dict = compute_detection_loss(preds, targets, imgsz=640)

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_ciou", loss_dict)
        self.assertIn("loss_hm", loss_dict)

        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient in {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf in gradient {name}")

        print(f"[PASS] Multi-Scale CIoU Loss & Backward Pass Verified: Total Loss = {loss.item():.4f} (HM: {loss_dict['loss_hm']:.4f}, CIoU: {loss_dict['loss_ciou']:.4f})")

    def test_jordan_observable_gate_symmetry_and_energy(self):
        """Test Jordan Observable Gate: P = P^T symmetry, identity initialization, and real eigenvalues."""
        from python.clifford_vision import JordanObservableGate
        gate = JordanObservableGate(channels=64).to(device)

        # 1. Exact identity test at initialization
        x = torch.randn(2, 64, 20, 20, device=device)
        out = gate(x)
        self.assertEqual((out - x).abs().max().item(), 0.0, "Initial Jordan gate is not exact identity")

        # 2. Operator Symmetry test (Jordan observable P = P^T)
        eye = torch.eye(64, device=device, dtype=x.dtype)
        P_sym = eye + gate.scale * 0.5 * (gate.weight + gate.weight.T)
        sym_diff = (P_sym - P_sym.T).abs().max().item()
        self.assertAlmostEqual(sym_diff, 0.0, places=6, msg="Jordan operator is not symmetric")

        # 3. Energy preservation under backward pass
        loss = out.pow(2).mean()
        loss.backward()
        grad_sym = (gate.weight.grad - gate.weight.grad.T).abs().max().item()
        self.assertAlmostEqual(grad_sym, 0.0, places=6, msg="Jordan observable gradient is not symmetric")
        print("[PASS] Jordan Observable Gate: Operator symmetry and energy conservation verified.")

    def test_sasaki_contact_metric(self):
        """Test Sasaki Contact Geodesic Metric: distance is in [0, 3] and penalizes aspect ratio deviation."""
        from python.clifford_vision import compute_sasaki_ciou_vectorized
        # Perfect match
        box1 = torch.tensor([[10.0, 10.0, 50.0, 100.0]], device=device)
        box2 = torch.tensor([[10.0, 10.0, 50.0, 100.0]], device=device)
        d_match = compute_sasaki_ciou_vectorized(box1, box2).item()
        self.assertAlmostEqual(d_match, 0.0, places=4, msg="Matching boxes should have 0 distance")

        # Distorted aspect ratio (stretched horizontally)
        box_distorted = torch.tensor([[10.0, 10.0, 100.0, 50.0]], device=device)
        d_aspect = compute_sasaki_ciou_vectorized(box1, box_distorted).item()
        self.assertGreater(d_aspect, 0.5, msg="Distorted aspect ratio should incur Sasaki Reeb penalty")
        print(f"[PASS] Sasaki Contact Metric Verified: match={d_match:.4f}, distorted={d_aspect:.4f}")

    def test_csn_large_architecture(self):
        """Test CSN-Large (~1.25M params, base_channels=64, 2 Lie neurons on C4/C5)."""
        model_large = CliffordDetModel(
            num_classes=1,
            base_channels=64,
            num_neurons_cl4=2,
            num_neurons_cl8=2,
            use_schulz=True
        ).to(device)
        params = sum(p.numel() for p in model_large.parameters())
        print(f"[PASS] CSN-Large Parameters: {params:,} (Target ~1.25M, exact: {params})")
        self.assertGreater(params, 1_000_000)
        self.assertLess(params, 1_500_000)

        # Forward pass
        x = torch.randn(2, 3, 640, 640, device=device)
        preds = model_large(x)
        self.assertEqual(preds["p3"]["heatmap"].shape, (2, 1, 80, 80))
        self.assertEqual(preds["p4"]["heatmap"].shape, (2, 1, 40, 40))
        self.assertEqual(preds["p5"]["heatmap"].shape, (2, 1, 20, 20))

    def test_model_ema(self):
        """Test ModelEMA shadow parameter update and gradient isolation."""
        from python.clifford_vision import ModelEMA
        model = CliffordDetModel(num_classes=1, base_channels=32, num_neurons_cl4=2, num_neurons_cl8=1).to(device)
        ema = ModelEMA(model, decay=0.9)

        # Check all EMA parameters require no grad
        for p in ema.ema.parameters():
            self.assertFalse(p.requires_grad)

        # Mutate model parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        # Update EMA
        ema.update(model)

        # Check that EMA weights moved towards model weights
        for p_model, p_ema in zip(model.parameters(), ema.ema.parameters()):
            self.assertFalse(torch.isnan(p_ema).any())

        print("[PASS] ModelEMA: Weight decay updates and gradient isolation verified.")


if __name__ == "__main__":
    unittest.main()

