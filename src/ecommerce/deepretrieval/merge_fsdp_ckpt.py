"""Merge verl FSDP sharded checkpoint into HuggingFace format.
Handles DTensor shards by extracting _local_tensor and concatenating across ranks.
"""
import argparse, os, torch
import torch.distributed.tensor
from collections import OrderedDict, defaultdict

def merge(ckpt_dir):
    actor_dir = os.path.join(ckpt_dir, "actor")
    hf_dir = os.path.join(actor_dir, "huggingface")
    shard_files = sorted([f for f in os.listdir(actor_dir) if f.startswith("model_world_size_")])
    print(f"Found {len(shard_files)} shards")
    
    # Collect local tensors from all ranks per key
    all_shards = defaultdict(list)  # key -> [(rank, local_tensor)]
    
    for rank, sf in enumerate(shard_files):
        print(f"Loading {sf}...")
        s = torch.load(os.path.join(actor_dir, sf), map_location="cpu", weights_only=False)
        for k, v in s.items():
            if hasattr(v, "_local_tensor"):
                local = v._local_tensor.clone()
                all_shards[k].append((rank, local))
            elif isinstance(v, torch.Tensor):
                all_shards[k].append((rank, v.clone()))
            else:
                all_shards[k].append((rank, v))
        del s
    
    # Merge: for sharded tensors, concat along dim 0; for replicated, take rank 0
    sd = OrderedDict()
    for k, rank_tensors in all_shards.items():
        rank_tensors.sort(key=lambda x: x[0])
        tensors = [t for _, t in rank_tensors]
        if not isinstance(tensors[0], torch.Tensor):
            sd[k] = tensors[0]
            continue
        # Check if all are the same (replicated) or different (sharded)
        if len(tensors) == 1:
            sd[k] = tensors[0]
        elif all(t.shape == tensors[0].shape and torch.equal(t, tensors[0]) for t in tensors[1:]):
            sd[k] = tensors[0]  # replicated
        else:
            # Sharded along dim 0
            sd[k] = torch.cat(tensors, dim=0)
    
    print(f"Keys: {len(sd)}")
    # Print a few key shapes for sanity check
    for i, (k, v) in enumerate(sd.items()):
        if i < 3 or i == len(sd) - 1:
            shape = v.shape if isinstance(v, torch.Tensor) else type(v)
            print(f"  {k}: {shape}")
    
    try:
        from safetensors.torch import save_file
        out = os.path.join(hf_dir, "model.safetensors")
        save_file(sd, out)
    except (ImportError, Exception) as e:
        print(f"safetensors failed ({e}), using pytorch format")
        out = os.path.join(hf_dir, "pytorch_model.bin")
        torch.save(sd, out)
    print(f"Saved {out} ({os.path.getsize(out)/1024/1024:.0f}MB)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("d")
    merge(p.parse_args().d)
