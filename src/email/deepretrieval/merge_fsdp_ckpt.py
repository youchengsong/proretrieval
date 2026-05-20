#!/usr/bin/env python3
"""Merge FSDP sharded checkpoints to HuggingFace format for inference.

Uses base model weight shapes as reference to correctly handle FSDP sharding
where equal-sized shards need concatenation (e.g., embeddings split along vocab dim).

Usage: python merge_fsdp_ckpt.py <checkpoint_dir>
Example: python merge_fsdp_ckpt.py checkpoints/global_step_100
"""
import glob
import os
import shutil
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL = "/root/autodl-tmp/models/Qwen3-4B"


def dtensor_to_numpy(t):
    if hasattr(t, "_local_tensor"):
        t = t._local_tensor
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy(), True
    return t.numpy(), False


def merge_fsdp_checkpoint(ckpt_dir):
    base_shapes = {}
    for f in sorted(glob.glob(f"{BASE_MODEL}/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                base_shapes[k] = sf.get_tensor(k).shape
    print(f"Base model: {len(base_shapes)} parameters")

    actor_dir = os.path.join(ckpt_dir, "actor")
    shard_files = sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(actor_dir, "*.pt")))
    if not shard_files:
        print(f"No shard files found in {actor_dir}")
        sys.exit(1)
    print(f"Found {len(shard_files)} shards")

    shards = []
    for f in shard_files:
        print(f"Loading {os.path.basename(f)}...")
        shards.append(torch.load(f, map_location="cpu", weights_only=False))

    merged = {}
    for key in shards[0]:
        if key == "lm_head.weight":
            continue
        expected_shape = base_shapes.get(key)
        if expected_shape is None:
            print(f"  WARNING: {key} not in base model, skipping")
            continue

        arrays = []
        is_bf16 = False
        for s in shards:
            arr, bf = dtensor_to_numpy(s[key])
            arrays.append(arr)
            is_bf16 = is_bf16 or bf

        if arrays[0].shape == tuple(expected_shape):
            arr = arrays[0]
        else:
            for dim in range(arrays[0].ndim):
                total = sum(a.shape[dim] for a in arrays)
                if total == expected_shape[dim]:
                    arr = np.concatenate(arrays, axis=dim)
                    break
            else:
                arr = arrays[0]

        t = torch.from_numpy(arr.copy())
        if is_bf16:
            t = t.to(torch.bfloat16)

        if tuple(t.shape) != tuple(expected_shape):
            print(f"  SHAPE MISMATCH: {key}: got {t.shape}, expected {expected_shape}")
        merged[key] = t

    del shards
    print(f"Merged {len(merged)} parameters")

    output_dir = os.path.join(actor_dir, "huggingface")
    os.makedirs(output_dir, exist_ok=True)

    for f in ["model.safetensors", "pytorch_model.bin"]:
        p = os.path.join(output_dir, f)
        if os.path.exists(p):
            os.remove(p)

    save_file(merged, os.path.join(output_dir, "model.safetensors"))
    print(f"Saved merged model to {output_dir}/model.safetensors")

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(BASE_MODEL, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"Merged checkpoint ready at {output_dir}")
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_dir>")
        sys.exit(1)
    merge_fsdp_checkpoint(sys.argv[1])
