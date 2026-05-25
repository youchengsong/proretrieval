#!/bin/bash
# Batch re-evaluation to add proper NDCG computation
# Run on ai4sci: cd ~/LocalSearch && bash scripts/batch_reeval_ndcg.sh

set -e
cd "$(dirname "$0")/.."

ECOM_EVAL="python src/eval_ecommerce.py"
ECOM_NOCASE="python src/eval_ecommerce_nocase.py"
EMAIL_EVAL="python src/eval_email.py"

ECOM_GT="data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
EMAIL_GT="data/email/test/queries_email_test_step9.jsonl"
ECOM_RES="data/ecommerce/result"
EMAIL_RES="data/email/result"

echo "=== Ecommerce: regular eval (case-sensitive) ==="
for pair in \
    "predictions_base.json:eval_base_gt_v2.json" \
    "predictions_sft_4b.json:eval_predictions_sft_4b_gt_v2.json" \
    "predictions_rl_step200.json:eval_rl_step200_gt_v2.json" \
    "predictions_rl_step300.json:eval_rl_step300_gt_v2.json" \
    "predictions_rl_step550.json:eval_rl_step550_gt_v2.json" \
    "predictions_rl_step1200.json:eval_rl_step1200_gt_v2.json" \
    "predictions_dapo_step420.json:eval_dapo_step420_gt_v2.json" \
; do
    pred="${pair%%:*}"
    out="${pair##*:}"
    echo "  $pred -> $out"
    $ECOM_EVAL --pred "$ECOM_RES/$pred" --gt "$ECOM_GT" --output "$ECOM_RES/$out"
done

echo ""
echo "=== Ecommerce: nocase eval (commercial models) ==="
for pair in \
    "predictions_gpt54_azure.json:eval_gpt54_azure_gt_v2_nocase.json" \
    "predictions_gpt54_thinking.json:eval_gpt54_thinking_gt_v2_nocase.json" \
    "predictions_claude_opus.json:eval_claude_opus_gt_v2_nocase.json" \
    "predictions_gemini_31pro_v2.json:eval_gemini_31pro_v2_gt_v2_nocase.json" \
    "predictions_gpt55_full.json:eval_predictions_gpt55_full.json" \
    "predictions_opus47.json:eval_predictions_opus47.json" \
    "predictions_fewshot_gpt55.json:eval_predictions_fewshot_gpt55.json" \
    "predictions_fewshot_opus47.json:eval_predictions_fewshot_opus47.json" \
; do
    pred="${pair%%:*}"
    out="${pair##*:}"
    echo "  $pred -> $out"
    $ECOM_NOCASE --pred "$ECOM_RES/$pred" --gt "$ECOM_GT" --output "$ECOM_RES/$out"
done

echo ""
echo "=== Email eval ==="
for pair in \
    "predictions_base_email.json:eval_predictions_base_email_step9_gt.json" \
    "predictions_sft_email.json:eval_predictions_sft_email_step9_gt.json" \
    "predictions_grpo_step843.json:eval_predictions_grpo_step843_step9_gt.json" \
    "predictions_dapo_step843.json:eval_predictions_dapo_step843_step9_gt.json" \
    "predictions_email_gpt54_list.json:eval_predictions_email_gpt54_list_step9_gt.json" \
    "predictions_email_gpt55_list.json:eval_predictions_email_gpt55_list_step9_gt.json" \
    "predictions_email_opus47_list.json:eval_predictions_email_opus47_list_step9_gt.json" \
; do
    pred="${pair%%:*}"
    out="${pair##*:}"
    echo "  $pred -> $out"
    $EMAIL_EVAL --pred "$EMAIL_RES/$pred" --gt "$EMAIL_GT" --output "$EMAIL_RES/$out"
done

echo ""
echo "=== 8B models (if present) ==="
for pair in \
    "predictions_base8b.json:eval_predictions_base8b.json" \
    "predictions_sft8b.json:eval_predictions_sft8b.json" \
    "predictions_grpo8b_step150.json:eval_predictions_grpo8b_step150.json" \
; do
    pred="${pair%%:*}"
    out="${pair##*:}"
    if [ -f "$ECOM_RES/$pred" ]; then
        echo "  $pred -> $out"
        $ECOM_EVAL --pred "$ECOM_RES/$pred" --gt "$ECOM_GT" --output "$ECOM_RES/$out"
    else
        echo "  SKIP $pred (not found)"
    fi
done

echo ""
echo "=== Done! ==="
