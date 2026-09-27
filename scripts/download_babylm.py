"""
Download BabyLM 2026 Strict-Small (10M words) and BabyLM dev evaluation sets.
Stores into data/babylm_10M/train/ and data/babylm_10M/dev/
"""

import os
import urllib.request
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "babylm_10M")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEV_DIR = os.path.join(DATA_DIR, "dev")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(DEV_DIR, exist_ok=True)

HF_TRAIN_BASE = "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/resolve/main"
HF_DEV_BASE = "https://huggingface.co/datasets/BabyLM-community/BabyLM-dev/resolve/main"

TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]

DEV_FILES = [
    "bnc_spoken.dev",
    "childes.dev",
    "gutenberg.dev",
    "open_subtitles.dev",
    "simple_wiki.dev",
    "switchboard.dev",
]


def download_file(url: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] {os.path.basename(out_path)} already exists ({os.path.getsize(out_path)} bytes)")
        return
    print(f"[DOWNLOAD] {url} -> {out_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"[DONE] {os.path.basename(out_path)} ({os.path.getsize(out_path)} bytes)")


def main():
    print("=" * 60)
    print("Downloading BabyLM 2026 Strict-Small Training Corpus (10M)")
    print("=" * 60)
    total_train_bytes = 0
    for fname in TRAIN_FILES:
        url = f"{HF_TRAIN_BASE}/{fname}"
        out = os.path.join(TRAIN_DIR, fname)
        download_file(url, out)
        total_train_bytes += os.path.getsize(out)

    print("\n" + "=" * 60)
    print("Downloading BabyLM Dev Evaluation Corpus")
    print("=" * 60)
    total_dev_bytes = 0
    for fname in DEV_FILES:
        url = f"{HF_DEV_BASE}/{fname}"
        out = os.path.join(DEV_DIR, fname)
        download_file(url, out)
        total_dev_bytes += os.path.getsize(out)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"Train dir: {TRAIN_DIR} ({total_train_bytes / (1024*1024):.2f} MB)")
    print(f"Dev dir:   {DEV_DIR} ({total_dev_bytes / (1024*1024):.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
