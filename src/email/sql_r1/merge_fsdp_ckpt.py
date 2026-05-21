#!/usr/bin/env python3
"""Merge verl FSDP sharded checkpoints into a single HuggingFace model.

Usage:
    python merge_fsdp_ckpt.py <checkpoint_dir>
    # e.g.: python merge_fsdp_ckpt.py checkpoints/global_step_100

Reads actor/model_world_size_*_rank_*.pt shards and merges into
actor/huggingface/model.safetensors
"""
import glob, os, sys
from collections import defaultdict
import torch
from torch.distributed.tensor import DTensor
from safetensors.torch import save_file


def to_local(t):
    if isinstance(t, DTensor):
        return t._local_tensor.contiguous()
    return t


def merge(ckpt_dir):
    actor_dir = os.path.join(ckpt_dir, "actor")
    if not os.path.isdir(actor_dir):
        print(f"ERROR: {actor_dir} not found")
        sys.exit(1)

    shards = sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(actor_dir, "model_*.safetensors")))
    if not shards:
        print("ERROR: No shards found")
        sys.exit(1)

    print(f"Found {len(shards)} shards")

    all_tensors = defaultdict(list)
    for path in shards:
        print(f"  Loading {os.path.basename(path)}...")
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, tensor in state.items():
            all_tensors[key].append(to_local(tensor))
        del state

    merged = {}
    for key, parts in all_tensors.items():
        if len(parts) == 1:
            merged[key] = parts[0]
        else:
            ref_shape = parts[0].shape
            for dim in range(len(ref_shape)):
                shapes = [p.shape[dim] for p in parts]
                if len(set(shapes)) > 1:
                    merged[key] = torch.cat(parts, dim=dim)
                    break
            else:
                if all(torch.equal(parts[0], p) for p in parts[1:]):
                    merged[key] = parts[0]
                else:
                    merged[key] = torch.cat(parts, dim=0)

    print(f"Merged {len(merged)} tensors")

    out_dir = os.path.join(actor_dir, "huggingface")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "model.safetensors")
    print(f"Saving merged model to {out_path} ...")
    save_file(merged, out_path)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Done: {len(merged)} tensors, {size_gb:.1f} GB")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_fsdp_ckpt.py <checkpoint_dir>")
        sys.exit(1)
    merge(sys.argv[1])
