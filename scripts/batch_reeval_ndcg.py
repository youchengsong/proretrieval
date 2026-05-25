#!/usr/bin/env python3
"""Batch re-evaluation: add proper NDCG@3/10 to all eval files.
Run on ai4sci: cd ~/LocalSearch-R1 && CUDA_VISIBLE_DEVICES=0 python scripts/batch_reeval_ndcg.py
"""
import json, os, sys, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/home/sunzhen/LocalSearch/data"

ECOM_EVAL = os.path.join(REPO, "src/eval_ecommerce.py")
ECOM_NOCASE = os.path.join(REPO, "src/eval_ecommerce_nocase.py")
EMAIL_EVAL = os.path.join(REPO, "src/eval_email.py")

ECOM_GT = f"{DATA}/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
ECOM_RES = f"{DATA}/ecommerce/result"
EMAIL_GT = f"{DATA}/email/test/queries_email_test_step9.jsonl"
EMAIL_RES = f"{DATA}/email/result"

ecom_regular = [
    ("predictions_base.json", "eval_base_gt_v2.json"),
    ("predictions_sft_4b.json", "eval_predictions_sft_4b_gt_v2.json"),
    ("predictions_rl_step200.json", "eval_rl_step200_gt_v2.json"),
    ("predictions_rl_step300.json", "eval_rl_step300_gt_v2.json"),
    ("predictions_rl_step550.json", "eval_rl_step550_gt_v2.json"),
    ("predictions_rl_step1200.json", "eval_rl_step1200_gt_v2.json"),
    ("predictions_dapo_step420.json", "eval_dapo_step420_gt_v2.json"),
    ("predictions_base8b.json", "eval_predictions_base8b.json"),
    ("predictions_sft8b.json", "eval_predictions_sft8b.json"),
    ("predictions_grpo8b_step150.json", "eval_predictions_grpo8b_step150.json"),
]

ecom_nocase = [
    ("predictions_gpt54_azure.json", "eval_gpt54_azure_gt_v2_nocase.json"),
    ("predictions_gpt54_thinking.json", "eval_gpt54_thinking_gt_v2_nocase.json"),
    ("predictions_claude_opus.json", "eval_claude_opus_gt_v2_nocase.json"),
    ("predictions_gemini_31pro_v2.json", "eval_gemini_31pro_v2_gt_v2_nocase.json"),
    ("predictions_gpt55_full.json", "eval_predictions_gpt55_full.json"),
    ("predictions_opus47.json", "eval_predictions_opus47.json"),
    ("predictions_fewshot_gpt55.json", "eval_predictions_fewshot_gpt55.json"),
    ("predictions_fewshot_opus47.json", "eval_predictions_fewshot_opus47.json"),
]

email_jobs = [
    ("predictions_base_email.json", "eval_predictions_base_email_step9_gt.json"),
    ("predictions_sft_email.json", "eval_predictions_sft_email_step9_gt.json"),
    ("predictions_grpo_step843.json", "eval_predictions_grpo_step843_step9_gt.json"),
    ("predictions_dapo_step843.json", "eval_predictions_dapo_step843_step9_gt.json"),
    ("predictions_email_gpt54_list.json", "eval_predictions_email_gpt54_list_step9_gt.json"),
    ("predictions_email_gpt55_list.json", "eval_predictions_email_gpt55_list_step9_gt.json"),
    ("predictions_email_opus47_list.json", "eval_predictions_email_opus47_list_step9_gt.json"),
]


def run_eval(script, pred, gt, output):
    if not os.path.exists(pred):
        print(f"  SKIP {os.path.basename(pred)} (not found)")
        return False
    print(f"  {os.path.basename(pred)} -> {os.path.basename(output)}", flush=True)
    cmd = [sys.executable, script, "--pred", pred, "--gt", gt, "--output", output]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"    ERROR: {r.stderr[-300:]}")
        return False
    for line in r.stdout.split("\n"):
        if "Hit@1:" in line or "ndcg" in line.lower():
            print(f"    {line.strip()}")
    return True


def patch_email_paths():
    """Temporarily patch eval_email.py paths for ai4sci."""
    with open(EMAIL_EVAL) as f:
        content = f.read()
    patched = content.replace(
        '/root/autodl-tmp/LocalSearch/data/email',
        '/home/sunzhen/LocalSearch/data/email'
    ).replace(
        '/root/autodl-tmp/models/Qwen3-Embedding-8B',
        '/home/sunzhen/models/Qwen3-Embedding-8B'
    )
    if patched != content:
        with open(EMAIL_EVAL, "w") as f:
            f.write(patched)
        print("  (patched eval_email.py paths for ai4sci)")
        return content
    return None


def main():
    total, ok = 0, 0

    print("=" * 60)
    print("Ecommerce: regular eval (case-sensitive)")
    print("=" * 60)
    for pred, out in ecom_regular:
        total += 1
        if run_eval(ECOM_EVAL, f"{ECOM_RES}/{pred}", ECOM_GT, f"{ECOM_RES}/{out}"):
            ok += 1

    print("\n" + "=" * 60)
    print("Ecommerce: nocase eval (commercial models)")
    print("=" * 60)
    for pred, out in ecom_nocase:
        total += 1
        if run_eval(ECOM_NOCASE, f"{ECOM_RES}/{pred}", ECOM_GT, f"{ECOM_RES}/{out}"):
            ok += 1

    print("\n" + "=" * 60)
    print("Email eval")
    print("=" * 60)
    original = patch_email_paths()
    for pred, out in email_jobs:
        total += 1
        if run_eval(EMAIL_EVAL, f"{EMAIL_RES}/{pred}", EMAIL_GT, f"{EMAIL_RES}/{out}"):
            ok += 1
    if original:
        with open(EMAIL_EVAL, "w") as f:
            f.write(original)
        print("  (restored eval_email.py paths)")

    print(f"\n{'=' * 60}")
    print(f"Done: {ok}/{total} evaluations completed")


if __name__ == "__main__":
    main()
