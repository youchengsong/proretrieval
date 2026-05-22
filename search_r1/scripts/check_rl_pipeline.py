"""Pre-flight check for Search-R1 RL pipeline - CPU only, no model loading."""
import json, sys, os
import pandas as pd
import numpy as np

print("=" * 60)
print("RL PIPELINE PRE-FLIGHT CHECK")
print("=" * 60)

errors = []
warnings = []

# ── CHECK 1: RL training data format ──
print("\n[CHECK 1] RL training data format")
for name, path in [
    ("ecommerce", "/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_train.parquet"),
    ("email", "/root/autodl-tmp/search_r1/rl_data/email_search_rl_train.parquet"),
]:
    if not os.path.exists(path):
        errors.append(f"RL data missing: {path}")
        continue
    df = pd.read_parquet(path)
    print(f"  {name}: {len(df)} samples, columns={list(df.columns)}")

    # Check required columns
    for col in ["prompt", "reward_model"]:
        if col not in df.columns:
            errors.append(f"{name}: missing column '{col}'")

    # Check prompt format
    row = df.iloc[0]
    prompt = row["prompt"]
    print(f"    prompt type: {type(prompt).__name__}, shape: {prompt.shape if hasattr(prompt, 'shape') else 'N/A'}")

    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        prompt = json.loads(prompt)

    if isinstance(prompt, list):
        print(f"    prompt msgs: {len(prompt)}, roles: {[m.get('role') for m in prompt]}")
        print(f"    user content preview: {prompt[-1]['content'][:100]}...")
    else:
        errors.append(f"{name}: prompt is not a list of messages after parsing: {type(prompt)}")

    # Check reward_model format
    rm_raw = row["reward_model"]
    print(f"    reward_model type: {type(rm_raw).__name__}")
    if isinstance(rm_raw, str):
        rm = json.loads(rm_raw)
    else:
        rm = rm_raw
    print(f"    reward_model keys: {list(rm.keys()) if isinstance(rm, dict) else 'N/A'}")

    gt = rm.get("ground_truth") if isinstance(rm, dict) else rm
    print(f"    ground_truth type: {type(gt).__name__}, sample: {str(gt)[:100]}")

# ── CHECK 2: Reward function correctness ──
print("\n[CHECK 2] Reward function")
sys.path.insert(0, "/root/autodl-tmp/scripts")
from reward_search_r1 import compute_score, extract_answer

# Test extract_answer
test_cases = [
    ("<think>reasoning</think>\n<search>query</search>\n<information>results</information>\n<think>more</think>\n<answer>B07GC9M3L6</answer>", "B07GC9M3L6"),
    ("<answer>ID123</answer>", "ID123"),
    ("no answer tags here", None),
    ("<think>x</think><answer>B07ABC</answer><think>y</think><answer>B07XYZ</answer>", "B07XYZ"),  # last match
]
for text, expected in test_cases:
    result = extract_answer(text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        errors.append(f"extract_answer failed: got {result}, expected {expected}")
    print(f"  extract_answer: {status} (got={result}, expected={expected})")

# Test compute_score with LIST ground truth (the actual format!)
gt_list = ["B07GC9M3L6", "B07F5S2KT3", "B07X2F9VM1"]
test_reward_cases = [
    # (solution, ground_truth, expected_score, description)
    ("<answer>B07GC9M3L6</answer>", gt_list, 1.0, "exact match with list GT"),
    ("<answer>B07F5S2KT3</answer>", gt_list, 1.0, "match 2nd item"),
    ("<answer>WRONG_ID</answer>", gt_list, 0.0, "no match"),
    ("no answer tag", gt_list, 0.0, "missing answer tag"),
    (None, gt_list, 0.0, "None solution"),
    ("<think>x</think><search>q</search><information>r</information><think>y</think><answer>B07X2F9VM1</answer>", gt_list, 1.0, "full chain with match"),
    ("<answer>B07GC9M3L6</answer>", "not_a_list", 0.0, "string GT as single ID - no match"),
    ("<answer>not_a_list</answer>", "not_a_list", 1.0, "string GT exact match"),
]

for solution, gt, expected, desc in test_reward_cases:
    score = compute_score("test", solution, gt, {})
    status = "PASS" if score == expected else "FAIL"
    if status == "FAIL":
        errors.append(f"compute_score failed: {desc} -> got {score}, expected {expected}")
    print(f"  compute_score: {status} | {desc} (got={score})")

# Test with actual parquet data
df = pd.read_parquet("/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_train.parquet")
rm = json.loads(df.iloc[0]["reward_model"])
real_gt = rm["ground_truth"]
# ground_truth can be dict {"target": [...]} or list [...]
if isinstance(real_gt, dict):
    first_id = real_gt["target"][0]
elif isinstance(real_gt, list):
    first_id = real_gt[0]
else:
    first_id = str(real_gt)
fake_solution = f"<think>found it</think><answer>{first_id}</answer>"
real_score = compute_score("test", fake_solution, real_gt, {})
print(f"  Real data test: score={real_score} (expected 1.0) for answer={first_id}")
if real_score != 1.0:
    errors.append(f"Real data reward test failed: got {real_score}")

# ── CHECK 3: verl data handling ──
print("\n[CHECK 3] verl data.return_raw_chat=True compatibility")
# When return_raw_chat=True, verl expects prompts to be chat messages
# Check if the prompts are in the right format
for name, path in [
    ("ecommerce", "/root/autodl-tmp/search_r1/rl_data/ecommerce_search_rl_train.parquet"),
    ("email", "/root/autodl-tmp/search_r1/rl_data/email_search_rl_train.parquet"),
]:
    df = pd.read_parquet(path)
    # Check 5 random samples
    for i in range(min(5, len(df))):
        prompt = df.iloc[i]["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if not isinstance(prompt, list):
            errors.append(f"{name} sample {i}: prompt not a list after parsing")
            continue
        for msg in prompt:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                errors.append(f"{name} sample {i}: invalid message format: {msg}")
                break
    print(f"  {name}: message format OK")

# ── CHECK 4: SFT checkpoint compatibility ──
print("\n[CHECK 4] SFT checkpoint files")
for name, path in [
    ("ecommerce_sft", "/root/autodl-tmp/search_r1/checkpoints/ecommerce_sft_v2/final"),
    ("email_sft", "/root/autodl-tmp/search_r1/checkpoints/email_sft_v2/final"),
]:
    if not os.path.exists(path):
        if "email" in name:
            warnings.append(f"{name}: not yet (still training)")
        else:
            errors.append(f"{name}: checkpoint missing at {path}")
        print(f"  {name}: {'PENDING' if 'email' in name else 'MISSING'}")
        continue
    files = os.listdir(path)
    has_model = any("model" in f and f.endswith(".safetensors") for f in files)
    has_config = "config.json" in files
    has_tokenizer = "tokenizer_config.json" in files
    print(f"  {name}: model={has_model}, config={has_config}, tokenizer={has_tokenizer}")
    if not all([has_model, has_config, has_tokenizer]):
        errors.append(f"{name}: incomplete checkpoint at {path}")

# ── CHECK 5: RL script configuration ──
print("\n[CHECK 5] RL script configuration")
with open("/root/autodl-tmp/scripts/run_search_rl.sh") as f:
    rl_script = f.read()

# Verify paths in script match actual files
checks = [
    ("ecommerce_sft_v2/final", "SFT ecom path"),
    ("email_sft_v2/final", "SFT email path"),
    ("ecommerce_search_rl_train.parquet", "RL ecom data"),
    ("email_search_rl_train.parquet", "RL email data"),
    ("reward_search_r1.py", "reward function"),
    ("data.return_raw_chat=True", "raw chat mode"),
    ("tensor_model_parallel_size=2", "tensor parallel"),
    ("n_gpus_per_node=2", "GPU count"),
]
for pattern, desc in checks:
    if pattern in rl_script:
        print(f"  {desc}: FOUND")
    else:
        errors.append(f"RL script missing: {desc} ({pattern})")
        print(f"  {desc}: MISSING!")

# ── CHECK 6: verl/vllm chat template handling ──
print("\n[CHECK 6] verl chat template handling")
# Check if verl uses apply_chat_template internally
verl_rl_dataset = "/root/autodl-tmp/SQL-R1/verl/utils/dataset/rl_dataset.py"
if os.path.exists(verl_rl_dataset):
    with open(verl_rl_dataset) as f:
        content = f.read()
    if "apply_chat_template" in content:
        print(f"  verl rl_dataset.py USES apply_chat_template!")
        print(f"  This could strip <search>/<information> tags!")
        # Find the relevant lines
        for i, line in enumerate(content.split("\n")):
            if "apply_chat_template" in line:
                print(f"    Line {i+1}: {line.strip()}")
        warnings.append("verl may use apply_chat_template which strips custom tags")
    else:
        print(f"  verl rl_dataset.py does NOT use apply_chat_template - OK")
else:
    print(f"  verl rl_dataset.py not found at expected path, checking alternatives...")
    import verl
    verl_path = os.path.dirname(verl.__file__)
    rl_ds_path = os.path.join(verl_path, "utils", "dataset", "rl_dataset.py")
    if os.path.exists(rl_ds_path):
        with open(rl_ds_path) as f:
            content = f.read()
        if "apply_chat_template" in content:
            print(f"  verl ({rl_ds_path}) USES apply_chat_template!")
            for i, line in enumerate(content.split("\n")):
                if "apply_chat_template" in line:
                    print(f"    Line {i+1}: {line.strip()}")
            warnings.append("verl may use apply_chat_template which strips custom tags")
        else:
            print(f"  verl does NOT use apply_chat_template - OK")
    else:
        warnings.append(f"Cannot locate verl rl_dataset.py to check template handling")

# ── SUMMARY ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")
if not errors and not warnings:
    print("\n✅ ALL CHECKS PASSED - RL pipeline ready")
elif not errors:
    print("\n✅ No blocking errors, but review warnings above")
else:
    print("\n❌ BLOCKING ERRORS found - fix before starting RL")
