"""
Calibrate YOLO26n on MOT16 Single-Frame Pedestrian Dataset.
"""

import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    print("====================================================================")
    print("CALIBRATING YOLO26n (Ultralytics) ON MOT16 PEDESTRIAN DATASET")
    print("====================================================================")
    
    yaml_path = Path("data/mot16/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"[ERROR] Missing dataset configuration at {yaml_path}")
        sys.exit(1)

    model = YOLO("yolo26n.pt")
    
    t0 = time.perf_counter()
    results = model.train(
        data=str(yaml_path),
        epochs=5,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/calibrate_mot16",
        name="yolo26n_calibrated",
        exist_ok=True,
        workers=4,
        save=True,
        val=True,
        plots=False,
        verbose=True
    )
    train_dur = time.perf_counter() - t0

    print("\n[SUCCESS] YOLO26n Calibration Completed in {:.1f}s".format(train_dur))
    best_weights = Path("runs/calibrate_mot16/yolo26n_calibrated/weights/best.pt")
    if best_weights.exists():
        print(f"[SUCCESS] Calibrated weights saved at: {best_weights}")

if __name__ == "__main__":
    main()
