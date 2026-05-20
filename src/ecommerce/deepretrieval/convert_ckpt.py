#!/usr/bin/env python3
"""Convert verl FSDP FULL_SHARD checkpoint to HuggingFace format.
Each rank stores 1/N of each parameter (sharded along dim 0). We load all ranks,
concatenate local tensors, and trim padding to original shape.
"""
import argparse, os, sys, torch, shutil
from collections import OrderedDict

def convert(ckpt_dir, base_model, output_dir=None):
    actor_dir = os.path.join(ckpt_dir, "actor")
    if output_dir is None:
        output_dir = os.path.join(actor_dir, "huggingface")
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if any(f.endswith(".safetensors") for f in files):
            print(f"Already converted: {output_dir}")
            return output_dir
        shutil.rmtree(output_dir)
    
    shard_files = sorted([f for f in os.listdir(actor_dir) if f.startswith("model_world_size_")])
    n_ranks = len(shard_files)
    if not shard_files:
        print("No FSDP shards found"); sys.exit(1)
    
    print(f"Loading {n_ranks} FSDP shards...")
    
    # Load all ranks
    all_shards = []
    for sf in shard_files:
        path = os.path.join(actor_dir, sf)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        all_shards.append(shard)
        print(f"  {sf}: {len(shard)} keys")
    
    # Get original shapes from DTensor metadata
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    
    # Build reference model to get expected shapes
    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    ref_state = ref_model.state_dict()
    
    # Merge: for each key, concatenate local tensors from all ranks along dim 0, then trim
    merged = OrderedDict()
    first_shard = all_shards[0]
    
    for key in first_shard.keys():
        locals_list = []
        for rank_shard in all_shards:
            t = rank_shard[key]
            # Extract local tensor from DTensor
            if hasattr(t, '_local_tensor'):
                local = t._local_tensor.detach().cpu()
            elif hasattr(t, 'to_local'):
                local = t.to_local().detach().cpu()
            else:
                local = t.detach().cpu()
            locals_list.append(local)
        
        if locals_list[0].dim() == 0:
            # Scalar - just take from rank 0
            merged[key] = locals_list[0].to(torch.bfloat16)
            continue
        
        # Concatenate along dim 0 (FSDP FULL_SHARD default)
        full = torch.cat(locals_list, dim=0)
        
        # Trim to original shape
        if key in ref_state:
            orig_shape = ref_state[key].shape
            if full.shape != orig_shape:
                # Trim padding
                slices = tuple(slice(0, s) for s in orig_shape)
                full = full[slices]
        
        merged[key] = full.to(torch.bfloat16)
    
    print(f"Merged {len(merged)} parameters")
    
    # Verify a few shapes
    for i, (k, v) in enumerate(merged.items()):
        if i < 3:
            expected = ref_state[k].shape if k in ref_state else "?"
            print(f"  {k}: {v.shape} (expected {expected})")
    
    # Load into model
    missing, unexpected = ref_model.load_state_dict(merged, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:3]}")
    
    os.makedirs(output_dir, exist_ok=True)
    ref_model.save_pretrained(output_dir, safe_serialization=True)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    # Verify
    final_files = [f for f in os.listdir(output_dir) if f.endswith('.safetensors')]
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in final_files)
    print(f"Saved to {output_dir} ({len(final_files)} safetensors, {total_size/1e9:.1f}GB)")
    
    del ref_model, merged, all_shards
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--base-model", default="/data/LLM_Models/Qwen3-4B")
    args = parser.parse_args()
    convert(args.ckpt, args.base_model)
