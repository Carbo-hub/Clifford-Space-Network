import os
import sys
import zipfile
import hashlib

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_zenodo():
    output_zip = "CSN_Zenodo_Release_v1.0.zip"
    print(f"Creating Zenodo Release Archive: {output_zip} ...")
    
    # Base files to include at root
    root_files = [
        ("docs/paper/clifford_space_network.pdf", "clifford_space_network.pdf"),
        ("README.md", "README.md"),
        ("README_ZENODO.md", "README_ZENODO.md"),
        ("LICENSE", "LICENSE"),
        (".zenodo.json", ".zenodo.json"),
        ("CMakeLists.txt", "CMakeLists.txt"),
        ("data/tinyshakespeare.txt", "data/tinyshakespeare.txt"),
    ]
    
    # Checkpoints to include
    ckpt_names = [
        "csn_v3_deep_unleash_best.pt",
        "csn_v3_mot16_transfer_best.pt",
        "csn_v3_coco_best.pt",
        "csn_lm_cl8_master.pt",
        "csn_large_mot16_best.pt",
        "csn_universal_mot16_best.pt",
    ]
    
    # Directories to walk and include (clean source trees only)
    source_dirs = [
        "python",
        "src",
        "include",
        "tests",
        "scripts",
        "formal",
        "docs",
        "csn_toolkit",
    ]
    
    files_added = []
    
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Add root files
        for src, arcname in root_files:
            if os.path.exists(src):
                zf.write(src, arcname)
                files_added.append((arcname, os.path.getsize(src)))
                print(f"  [+] {arcname} ({os.path.getsize(src) / 1024:.1f} KB)")
            else:
                print(f"  [!] Missing expected file: {src}")
                
        # 2. Add checkpoints
        for ckpt in ckpt_names:
            src = os.path.join("checkpoints", ckpt)
            arcname = f"checkpoints/{ckpt}"
            if os.path.exists(src):
                zf.write(src, arcname)
                files_added.append((arcname, os.path.getsize(src)))
                print(f"  [+] {arcname} ({os.path.getsize(src) / (1024 * 1024):.2f} MB)")
            else:
                print(f"  [!] Missing checkpoint: {src}")
                
        # 3. Add clean source directories
        for sdir in source_dirs:
            if not os.path.exists(sdir):
                continue
            for root, dirs, files in os.walk(sdir):
                # Filter out unwanted directories
                dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache', 'build', '.egg-info', '.git')]
                for file in files:
                    if file.endswith(('.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', '.tmp')):
                        continue
                    # Also skip duplicate large weights or logs if in those dirs
                    if file.endswith(('.pt', '.pth', '.bin', '.tar.gz', '.zip')):
                        # Don't include other zips or loose weights inside docs/tests
                        continue
                    src_path = os.path.join(root, file)
                    arcname = os.path.relpath(src_path, ".").replace("\\", "/")
                    zf.write(src_path, arcname)
                    files_added.append((arcname, os.path.getsize(src_path)))

    zip_size = os.path.getsize(output_zip)
    zip_sha = get_sha256(output_zip)
    print("=" * 80)
    print(f"Archive successfully generated: {output_zip}")
    print(f"Total files packed: {len(files_added)}")
    print(f"Total archive size: {zip_size / (1024 * 1024):.2f} MB ({zip_size:,} bytes)")
    print(f"SHA-256 Checksum:  {zip_sha}")
    print("=" * 80)
    
    # Save a verification manifest alongside the zip
    manifest_path = "CSN_Zenodo_Release_v1.0_MANIFEST.txt"
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        mf.write(f"Archive: {output_zip}\n")
        mf.write(f"Size: {zip_size} bytes ({zip_size / (1024 * 1024):.2f} MB)\n")
        mf.write(f"SHA256: {zip_sha}\n\n")
        mf.write("Contained Files:\n")
        for arcname, size in sorted(files_added):
            mf.write(f"  {size:10d} bytes | {arcname}\n")
    print(f"Manifest written to: {manifest_path}")

if __name__ == '__main__':
    package_zenodo()
