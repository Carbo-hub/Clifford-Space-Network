"""
Download and Prepare MOT16 Dataset for Single-Frame YOLO and Clifford Space Network (CSN).
Source: Hugging Face mirror (AbdoTW/MOT16-17-20).
"""

import os
import sys
import shutil
import zipfile
import configparser
import urllib.request
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path("data/mot16").resolve()
RAW_DIR = Path("data/mot16_raw").resolve()
ZIP_PATH = Path("data/MOT16-train.zip").resolve()
URL = "https://huggingface.co/datasets/AbdoTW/MOT16-17-20/resolve/main/MOT16-train.zip"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_mot16():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "train").exists():
        print(f"[INFO] Raw MOT16 train directory already exists at: {RAW_DIR / 'train'}")
        return

    if not ZIP_PATH.exists():
        print(f"[INFO] Downloading MOT16-train.zip from {URL}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, 'wb') as f:
            total_size = int(resp.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="MOT16-train.zip") as pbar:
                while True:
                    chunk = resp.read(1024 * 1024) # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"[SUCCESS] Download completed: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)")

    print(f"[INFO] Extracting {ZIP_PATH} to {RAW_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print(f"[SUCCESS] Extraction complete at {RAW_DIR}.")


def parse_seq_info(seq_dir: Path):
    for ini_path in seq_dir.glob("*info.ini"):
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            w = int(config['Sequence']['imWidth'])
            h = int(config['Sequence']['imHeight'])
            return w, h
        except Exception:
            pass
    return 1920, 1080


def convert_mot16_to_yolo():
    print("[INFO] Converting MOT16 to Single-Frame YOLO format...")
    
    # Locate train root
    train_root = None
    for p in RAW_DIR.rglob("MOT16-02"):
        train_root = p.parent
        break

    if train_root is None:
        raise FileNotFoundError(f"Could not locate MOT16 sequences inside {RAW_DIR}")

    print(f"[INFO] Discovered sequences root at: {train_root}")
    seq_dirs = sorted([d for d in train_root.iterdir() if d.is_dir() and (d / "img1").exists()])
    print(f"[INFO] Found {len(seq_dirs)} active sequences: {[d.name for d in seq_dirs]}")

    # Set MOT16-04 as the validation sequence, others as train
    val_seq_names = {"MOT16-04"}

    train_img_dir = DATA_DIR / "images" / "train"
    train_lbl_dir = DATA_DIR / "labels" / "train"
    val_img_dir = DATA_DIR / "images" / "val"
    val_lbl_dir = DATA_DIR / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_train_frames = 0
    total_val_frames = 0

    for seq in seq_dirs:
        seq_name = seq.name
        is_val = seq_name in val_seq_names
        img_out_dir = val_img_dir if is_val else train_img_dir
        lbl_out_dir = val_lbl_dir if is_val else train_lbl_dir

        img1_dir = seq / "img1"
        gt_file = seq / "gt" / "gt.txt"
        img_w, img_h = parse_seq_info(seq)

        print(f"[INFO] Processing {seq_name} ({img_w}x{img_h}) -> {'VAL' if is_val else 'TRAIN'}")

        # Parse annotations: class 1 = Pedestrian, visibility > 0.25
        annotations = {}
        if gt_file.exists():
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue
                    frame_idx = int(parts[0])
                    class_id = int(parts[7])
                    vis = float(parts[8]) if len(parts) > 8 else 1.0

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
                        # Class 0: pedestrian
                        annotations[frame_idx].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Copy or symlink frames
        for img_path in sorted(img1_dir.glob("*.jpg")):
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                continue

            target_name = f"{seq_name}_{frame_idx:06d}"
            target_img = img_out_dir / f"{target_name}.jpg"
            target_lbl = lbl_out_dir / f"{target_name}.txt"

            if not target_img.exists():
                shutil.copy2(img_path, target_img)

            with open(target_lbl, "w") as lf:
                if frame_idx in annotations:
                    lf.write("\n".join(annotations[frame_idx]) + "\n")

            if is_val:
                total_val_frames += 1
            else:
                total_train_frames += 1

    print(f"\n[SUCCESS] Dataset Prepared:")
    print(f"  Training Frames:   {total_train_frames}")
    print(f"  Validation Frames: {total_val_frames}")

    # Create data.yaml
    yaml_content = f"""path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: pedestrian
"""
    with open(DATA_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"[SUCCESS] Configuration written to: {DATA_DIR / 'data.yaml'}")


if __name__ == "__main__":
    download_mot16()
    convert_mot16_to_yolo()
