#!/usr/bin/env python3
"""Merge verl FSDP-sharded DTensor checkpoint into a HuggingFace model."""
import argparse
import torch
import os
import shutil
from collections import OrderedDict

def merge_fsdp_checkpoint(ckpt_dir, base_model_dir, output_dir):
    actor_dir = os.path.join(ckpt_dir, "actor")
    hf_dir = os.path.join(actor_dir, "huggingface")

    shard_files = sorted([
        f for f in os.listdir(actor_dir)
        if f.startswith("model_world_size_") and f.endswith(".pt")
    ])
    print(f"Found {len(shard_files)} shards: {shard_files}")

    shards = []
    for sf in shard_files:
        path = os.path.join(actor_dir, sf)
        print(f"Loading {sf}...")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shards.append(shard)

    first_shard = shards[0]

    # Merge DTensors
    merged_state = OrderedDict()
    for key in first_shard.keys():
        tensors = []
        for shard in shards:
            t = shard[key]
            if hasattr(t, '_local_tensor'):
                local = t._local_tensor
            else:
                local = t
            tensors.append(local)

        dt = first_shard[key]
        if hasattr(dt, 'placements') and len(dt.placements) > 0:
            from torch.distributed.tensor import Replicate, Shard as DTShard
            placement = dt.placements[0]
            if isinstance(placement, Replicate):
                merged_state[key] = tensors[0].clone()
            elif isinstance(placement, DTShard):
                merged_state[key] = torch.cat(tensors, dim=placement.dim)
            else:
                merged_state[key] = tensors[0].clone()
        else:
            merged_state[key] = tensors[0].clone()

    print(f"Merged {len(merged_state)} parameters")
    for i, (k, v) in enumerate(merged_state.items()):
        if i < 3:
            print(f"  {k}: {v.shape} {v.dtype}")

    # Load base model architecture from SFT checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model architecture from {base_model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, dtype=torch.bfloat16
    )

    print("Loading merged weights...")
    result = model.load_state_dict(merged_state, strict=False)
    if result.missing_keys:
        print(f"Missing keys: {len(result.missing_keys)}")
        for k in result.missing_keys[:3]:
            print(f"  {k}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {len(result.unexpected_keys)}")
        for k in result.unexpected_keys[:3]:
            print(f"  {k}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    merge_fsdp_checkpoint(args.ckpt_dir, args.base_model_dir, args.output_dir)
