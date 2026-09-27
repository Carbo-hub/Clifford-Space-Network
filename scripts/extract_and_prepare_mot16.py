"""
Fast in-memory extraction of MOT16-02 sequence directly from downloaded zip chunk.
Creates clean Single-Frame YOLO pedestrian dataset (train/val split) in data/mot16.
"""

import os
import sys
import struct
import zlib
from pathlib import Path

DATA_DIR = Path("data/mot16").resolve()
RAW_DIR = Path("data/mot16_raw").resolve()
ZIP_PATH = Path("data/MOT16-train.zip").resolve()


def extract_mot16_02():
    print("====================================================================")
    print("FAST EXTRACTION & PREPARATION OF MOT16-02 BENCHMARK")
    print("====================================================================")

    if not ZIP_PATH.exists():
        print(f"[ERROR] Missing {ZIP_PATH}")
        sys.exit(1)

    print(f"[INFO] Reading zip chunk from {ZIP_PATH}...")
    with open(ZIP_PATH, 'rb') as f:
        raw = f.read()

    seq_dir = RAW_DIR / "MOT16-02"
    img1_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img1_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pos = 0
    extracted_imgs = 0
    gt_content = None

    while True:
        idx = raw.find(b'PK\x03\x04', pos)
        if idx == -1:
            break
        if idx + 30 <= len(raw):
            comp_method, mod_time, mod_date, crc32, comp_size, uncomp_size, fn_len, extra_len = struct.unpack(
                '<HHHIIIHH', raw[idx+8:idx+30]
            )
            fn = raw[idx+30:idx+30+fn_len].decode('utf-8', errors='ignore')
            data_start = idx + 30 + fn_len + extra_len
            data_end = data_start + comp_size

            if 'MOT16-02' in fn and comp_size > 0 and data_end <= len(raw):
                blob = raw[data_start:data_end]
                content = zlib.decompress(blob, -15) if comp_method == 8 else blob

                if fn.endswith('.jpg'):
                    out_path = img1_dir / Path(fn).name
                    with open(out_path, 'wb') as out_f:
                        out_f.write(content)
                    extracted_imgs += 1
                elif 'gt.txt' in fn:
                    gt_content = content.decode('utf-8', errors='ignore')
                    with open(gt_dir / "gt.txt", 'w', encoding='utf-8') as out_f:
                        out_f.write(gt_content)
                elif 'seqinfo.ini' in fn:
                    with open(seq_dir / "seqinfo.ini", 'wb') as out_f:
                        out_f.write(content)

        pos = idx + 4

    print(f"[SUCCESS] Extracted {extracted_imgs} frames and annotations to {seq_dir}")

    # Convert to single-frame YOLO format
    print("\n[INFO] Converting to YOLO Single-Frame Pedestrian format...")
    train_img_dir = DATA_DIR / "images" / "train"
    train_lbl_dir = DATA_DIR / "labels" / "train"
    val_img_dir = DATA_DIR / "images" / "val"
    val_lbl_dir = DATA_DIR / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img_w, img_h = 1920, 1080

    annotations = {}
    if gt_content:
        for line in gt_content.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            class_id = int(parts[7])
            vis = float(parts[8]) if len(parts) > 8 else 1.0

            # Class 1 = Pedestrian
            if class_id == 1 and vis > 0.25:
                x_left = float(parts[2])
                y_top = float(parts[3])
                w_box = float(parts[4])
                h_box = float(parts[5])

                cx = min(max((x_left + w_box / 2.0) / img_w, 0.0), 1.0)
                cy = min(max((y_top + h_box / 2.0) / img_h, 0.0), 1.0)
                nw = min(max(w_box / img_w, 0.0), 1.0)
                nh = min(max(h_box / img_h, 0.0), 1.0)

                if frame_idx not in annotations:
                    annotations[frame_idx] = []
                annotations[frame_idx].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    all_jpgs = sorted(list(img1_dir.glob("*.jpg")))
    train_count = 0
    val_count = 0

    # Frames 1-450 train, 451-600 val
    for p in all_jpgs:
        try:
            frame_idx = int(p.stem)
        except ValueError:
            continue

        is_val = frame_idx > 450
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        target_img = target_img_dir / p.name
        target_lbl = target_lbl_dir / f"{p.stem}.txt"

        if not target_img.exists():
            import shutil
            shutil.copy2(p, target_img)

        with open(target_lbl, "w") as f:
            if frame_idx in annotations:
                f.write("\n".join(annotations[frame_idx]) + "\n")

        if is_val:
            val_count += 1
        else:
            train_count += 1

    print(f"[SUCCESS] Dataset Split Complete:")
    print(f"  Training Frames:   {train_count} (frames 1-450)")
    print(f"  Validation Frames: {val_count} (frames 451-600)")

    yaml_content = f"""path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: pedestrian
"""
    with open(DATA_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] data.yaml saved at: {DATA_DIR / 'data.yaml'}")


if __name__ == "__main__":
    extract_mot16_02()
