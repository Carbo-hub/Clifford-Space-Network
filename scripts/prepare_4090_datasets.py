"""
Automatic Downloader & Tokenizer Preprocessor for:
1. BabyLM 100M (BabyLM-2026-Strict)
2. TinyStories (roneneldan/TinyStories)

Optimized for high-throughput streaming and pre-tokenization.
"""

import os
import urllib.request
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# BabyLM 100M official files (BabyLM-2026-Strict)
BABYLM_100M_TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
HF_BABYLM_100M_BASE = "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict/resolve/main"

# TinyStories Hugging Face parquet URLs (or text splits)
HF_TINYSTORIES_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TINYSTORIES_FILES = [
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt"
]

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[SKIP] {dest_path.name} already exists ({dest_path.stat().st_size / 1e6:.2f} MB)")
        return
    print(f"[DOWNLOAD] {url} -> {dest_path} ...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        downloaded = 0
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            print(f"\r  Downloaded: {downloaded / 1e6:.1f} MB", end="", flush=True)
    print(f"\n[DONE] {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")

def download_babylm_100m():
    target_dir = DATA_DIR / "babylm_100M" / "train"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("Downloading BabyLM 2026 Strict (100M) Corpus")
    print("=" * 60)
    for fname in BABYLM_100M_TRAIN_FILES:
        url = f"{HF_BABYLM_100M_BASE}/{fname}"
        out = target_dir / fname
        try:
            download_file(url, out)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")

def download_tinystories():
    target_dir = DATA_DIR / "tinystories"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("Downloading TinyStories-V2 (GPT-4) Corpus")
    print("=" * 60)
    for fname in TINYSTORIES_FILES:
        url = f"{HF_TINYSTORIES_BASE}/{fname}"
        out = target_dir / fname
        try:
            download_file(url, out)
        except Exception as e:
            print(f"Direct text download for {fname}: {e}")
            print(f"Note: TinyStories can also be loaded via `datasets.load_dataset('roneneldan/TinyStories')`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["babylm", "tinystories", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ["babylm", "all"]:
        download_babylm_100m()
    if args.dataset in ["tinystories", "all"]:
        download_tinystories()
