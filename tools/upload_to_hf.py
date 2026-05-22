#!/usr/bin/env python3
"""
Sync /data/youchengsong/DSL-R1/data → HuggingFace dataset repo chengsong3/LR_R1.

- Uploads new/changed files (detected by file size diff; use --force to re-upload all)
- Deletes files on HF that no longer exist locally
- Excludes intermediate _tmp_*.jsonl files
"""

import fnmatch
import argparse
from pathlib import Path
from huggingface_hub import HfApi

TOKEN    = "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"
REPO_ID  = "chengsong3/LR_R1"
DATA_DIR = Path("/data/youchengsong/DSL-R1/data")
ENDPOINT = "https://hf-mirror.com"

IGNORE_PATTERNS = ["_tmp_*.jsonl"]

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Re-upload all files even if size matches")
parser.add_argument("--dry-run", action="store_true", help="Print what would be done without uploading")
args = parser.parse_args()

api = HfApi(endpoint=ENDPOINT)

# Ensure repo exists
try:
    api.repo_info(repo_id=REPO_ID, repo_type="dataset", token=TOKEN)
    print(f"Repo {REPO_ID} exists.")
except Exception:
    print(f"Creating repo {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", token=TOKEN, private=False)
    print("Repo created.")

# ── Collect local files ────────────────────────────────────────────────────────
local_files: dict[str, Path] = {}  # repo_path → local Path
for f in DATA_DIR.rglob("*"):
    if not f.is_file():
        continue
    if any(fnmatch.fnmatch(f.name, pat) for pat in IGNORE_PATTERNS):
        continue
    repo_path = str(f.relative_to(DATA_DIR))
    local_files[repo_path] = f

# ── Collect HF files ───────────────────────────────────────────────────────────
print("Fetching file list from HuggingFace ...")
hf_files: dict[str, int] = {}  # repo_path → size in bytes
try:
    for item in api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", token=TOKEN, recursive=True):
        if item.type == "file":  # RepoFile has .rfilename and .size
            hf_files[item.rfilename] = item.size or 0
except Exception as e:
    print(f"Warning: could not list HF files: {e}")

print(f"Local: {len(local_files)} files | HF: {len(hf_files)} files\n")

# ── Upload new / changed files ─────────────────────────────────────────────────
to_upload = []
for repo_path, local_path in sorted(local_files.items()):
    local_size = local_path.stat().st_size
    hf_size    = hf_files.get(repo_path)
    if args.force or hf_size is None or hf_size != local_size:
        reason = "new" if hf_size is None else ("forced" if args.force else f"size {hf_size}→{local_size}")
        to_upload.append((repo_path, local_path, reason))

print(f"Files to upload: {len(to_upload)}")
for i, (repo_path, local_path, reason) in enumerate(to_upload, 1):
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"[{i}/{len(to_upload)}] {repo_path} ({size_mb:.1f} MB, {reason})")
    if args.dry_run:
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=TOKEN,
        )
        print(f"  OK")
    except Exception as e:
        print(f"  FAILED: {e}")

# ── Delete stale files from HF ────────────────────────────────────────────────
to_delete = [rp for rp in hf_files if rp not in local_files]
if to_delete:
    print(f"\nFiles to delete from HF: {len(to_delete)}")
    for rp in to_delete:
        print(f"  DELETE {rp}")
        if args.dry_run:
            continue
        try:
            api.delete_file(path_in_repo=rp, repo_id=REPO_ID, repo_type="dataset", token=TOKEN)
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
else:
    print("\nNo stale files to delete from HF.")

print("\nDone!")
