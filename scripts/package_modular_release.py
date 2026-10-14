import os
import sys
import shutil
import zipfile
import hashlib

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_modular():
    print("=" * 80)
    print("  BUILDING MODULAR ZENODO DEPOSIT BUNDLE")
    print("=" * 80)

    bundle_dir = "zenodo_bundle"
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    # 1. Standalone PDF Paper
    pdf_src = os.path.join("docs", "paper", "clifford_space_network.pdf")
    pdf_dst = os.path.join(bundle_dir, "clifford_space_network.pdf")
    shutil.copy2(pdf_src, pdf_dst)
    print(f"  [1/4] PDF copied: {pdf_dst} ({os.path.getsize(pdf_dst) / 1024:.1f} KB)")

    # 2. Pure Source Code Zip (no heavy weights, instant upload <1.5 MB)
    src_zip_path = os.path.join(bundle_dir, "CSN_Source_Code_v1.0.zip")
    root_files = [
        ("README.md", "README.md"),
        ("README_ZENODO.md", "README_ZENODO.md"),
        ("LICENSE", "LICENSE"),
        (".zenodo.json", ".zenodo.json"),
        ("CMakeLists.txt", "CMakeLists.txt"),
        ("data/tinyshakespeare.txt", "data/tinyshakespeare.txt"),
    ]
    source_dirs = ["python", "src", "include", "tests", "scripts", "formal", "docs", "csn_toolkit"]

    with zipfile.ZipFile(src_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in root_files:
            if os.path.exists(src):
                zf.write(src, arcname)
        for sdir in source_dirs:
            if not os.path.exists(sdir):
                continue
            for root, dirs, files in os.walk(sdir):
                dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache', 'build', '.git')]
                for file in files:
                    if file.endswith(('.pyc', '.pt', '.pth', '.zip', '.tmp')):
                        continue
                    src_p = os.path.join(root, file)
                    arcname = os.path.relpath(src_p, ".").replace("\\", "/")
                    zf.write(src_p, arcname)

    print(f"  [2/4] Source Code Archive: {src_zip_path} ({os.path.getsize(src_zip_path) / 1024:.1f} KB)")

    # 3. Individual Checkpoints directly in bundle folder for individual upload
    checkpoints = [
        ("checkpoints/csn_lm_cl8_master.pt", "csn_lm_cl8_master.pt"),
        ("checkpoints/csn_v3_mot16_transfer_best.pt", "csn_v3_mot16_transfer_best.pt"),
        ("checkpoints/csn_v3_deep_unleash_best.pt", "csn_v3_deep_unleash_best.pt"),
        ("checkpoints/csn_v3_coco_best.pt", "csn_v3_coco_best.pt"),
    ]
    for src, fname in checkpoints:
        if os.path.exists(src):
            dst = os.path.join(bundle_dir, fname)
            shutil.copy2(src, dst)
            print(f"  [3/4] Checkpoint staged: {fname} ({os.path.getsize(dst) / (1024*1024):.2f} MB)")

    # 4. Also create modular zip archives for users who want zipped models
    # A) Vision models zip (split in 2 parts if preferred or single vision zip)
    vision_zip_path = os.path.join(bundle_dir, "CSN_Vision_Models_v1.0.zip")
    with zipfile.ZipFile(vision_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in ["csn_v3_deep_unleash_best.pt", "csn_v3_mot16_transfer_best.pt", "csn_v3_coco_best.pt"]:
            p = os.path.join("checkpoints", f)
            if os.path.exists(p):
                zf.write(p, f"checkpoints/{f}")
    print(f"  [4/4] Vision Models Archive: {vision_zip_path} ({os.path.getsize(vision_zip_path) / (1024*1024):.2f} MB)")

    # Language model zip
    lm_zip_path = os.path.join(bundle_dir, "CSN_Language_Models_v1.0.zip")
    with zipfile.ZipFile(lm_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        p = os.path.join("checkpoints", "csn_lm_cl8_master.pt")
        if os.path.exists(p):
            zf.write(p, "checkpoints/csn_lm_cl8_master.pt")

    # Generate Manifest
    manifest_path = os.path.join(bundle_dir, "MODULAR_UPLOAD_MANIFEST.txt")
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        mf.write("ZENODO MODULAR UPLOAD BUNDLE - RECOMMENDED UPLOAD BREAKDOWN\n")
        mf.write("=" * 80 + "\n\n")
        mf.write("You can upload these files individually to Zenodo (avoiding large single-zip timeout):\n\n")
        for f in sorted(os.listdir(bundle_dir)):
            if f.endswith(".txt"):
                continue
            fp = os.path.join(bundle_dir, f)
            sz = os.path.getsize(fp)
            sha = get_sha256(fp)
            mf.write(f"- {f:<35} | {sz / (1024*1024):6.2f} MB | SHA256: {sha}\n")

    print("\n" + "=" * 80)
    print(f"Bundle successfully created in folder: '{bundle_dir}/'")
    for f in sorted(os.listdir(bundle_dir)):
        fp = os.path.join(bundle_dir, f)
        print(f"  * {f:<35} : {os.path.getsize(fp) / (1024*1024):.2f} MB")
    print("=" * 80)

if __name__ == '__main__':
    package_modular()
