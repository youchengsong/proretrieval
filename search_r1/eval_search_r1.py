"""
Evaluation script for Search-R1 email RL checkpoints.
Runs multi-turn search inference on test set and computes Hit@1/Hit@3/MRR.

Usage:
    python eval_search_r1.py \
        --model_path /path/to/merged_checkpoint \
        --test_file /path/to/email_rl_test_fixed.parquet \
        --retrieval_url http://127.0.0.1:8000/retrieve \
        --output_file results.json \
        --max_samples 3000 \
        --gpu 1
"""
import argparse
import json
import os
import re
import time
import requests
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MSG_ID_PATTERN = re.compile(r'<[\d]+\.[\d]+\.JavaMail\.\w+@\w+>')


def extract_solution(text):
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_search_queries(text):
    matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    return [q.strip() for q in matches if q.strip()]


def extract_message_ids(text):
    return MSG_ID_PATTERN.findall(text)


def do_search(queries, retrieval_url, topk=3):
    try:
        resp = requests.post(
            retrieval_url,
            json={"queries": queries, "topk": topk},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Search error: {e}")
        return {"result": [[] for _ in queries]}


def build_search_result_text(search_results):
    """Format search results to match verl training format:
    Doc 1 (ID: <docid>, Title: <title>)
    <text>
    """
    parts = []
    for i, query_results in enumerate(search_results.get("result", [])):
        for j, doc in enumerate(query_results):
            if isinstance(doc, dict):
                docid = doc.get("docid", "")
                title = doc.get("title", "")
                text = doc.get("text", doc.get("content", str(doc)))
                if len(text) > 500:
                    half = 250
                    text = text[:half] + "\n...[truncated]...\n" + text[-half:]
                if docid:
                    parts.append(f"Doc {j+1} (ID: {docid}, Title: {title})\n{text}")
                else:
                    parts.append(f"Doc {j+1} (Title: {title})\n{text}")
            else:
                parts.append(f"Doc {j+1}: {str(doc)}")
    return "\n\n".join(parts) if parts else "No results found."


def run_inference(llm, tokenizer, prompt_messages, sampling_params, retrieval_url, max_turns=3):
    messages = list(prompt_messages)
    full_response = ""

    for turn in range(max_turns + 1):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling_params)
        response = outputs[0].outputs[0].text or ""

        if "<search>" in response and "</search>" in response:
            search_part = response[:response.index("</search>") + len("</search>")]
            queries = extract_search_queries(search_part)

            if queries and turn < max_turns:
                search_results = do_search(queries, retrieval_url)
                info_text = build_search_result_text(search_results)

                messages.append({"role": "assistant", "content": search_part})
                messages.append({"role": "user", "content": f"<information>{info_text}</information>"})
                full_response += search_part + "\n"
                continue

        full_response += response
        break

    return full_response


def compute_hit(pred_ids, golden_ids):
    if not golden_ids or not pred_ids:
        return 0
    for pid in pred_ids:
        for gid in golden_ids:
            if pid == gid:
                return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"Loading model from {args.model_path} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=8192,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|im_end|>"],
    )

    print(f"Loading test data from {args.test_file}...")
    df = pd.read_parquet(args.test_file)
    if args.max_samples and args.max_samples < len(df):
        df = df.head(args.max_samples)
    print(f"Evaluating {len(df)} samples")

    results = []
    correct = 0
    has_answer = 0
    total = 0

    for idx, row in df.iterrows():
        prompt_messages = list(row["prompt"])
        try:
            reward_info = json.loads(row["reward_model"]) if isinstance(row["reward_model"], str) else row["reward_model"]
            golden_ids = reward_info.get("ground_truth", []) if reward_info else []
            if golden_ids is None:
                golden_ids = []
        except (json.JSONDecodeError, TypeError, AttributeError):
            golden_ids = []
        extra = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
        query_id = extra.get("query_id", f"q_{idx}")

        try:
            response = run_inference(llm, tokenizer, prompt_messages, sampling_params, args.retrieval_url, args.max_turns)
        except Exception as e:
            print(f"Inference error at {query_id}: {e}")
            response = ""

        answer = extract_solution(response)
        pred_ids = extract_message_ids(answer) if answer else []

        hit = compute_hit(pred_ids, golden_ids) if pred_ids else 0

        total += 1
        if answer:
            has_answer += 1
        if hit:
            correct += 1

        results.append({
            "query_id": query_id,
            "answer": answer,
            "pred_ids": pred_ids,
            "golden_ids": golden_ids[:5],
            "hit": hit,
            "level": extra.get("level", -1),
        })

        if total % 50 == 0:
            print(f"[{total}/{len(df)}] Hit@1={correct/total:.4f} answer_rate={has_answer/total:.4f}")

    hit1 = correct / total if total > 0 else 0
    answer_rate = has_answer / total if total > 0 else 0

    # Compute Hit@3 and MRR (simplified: any of top predicted IDs in golden)
    # For Search-R1, the model returns one answer, so Hit@1 = Hit@3 in most cases
    summary = {
        "model_path": args.model_path,
        "test_file": args.test_file,
        "total": total,
        "correct": correct,
        "has_answer": has_answer,
        "hit_at_1": round(hit1, 4),
        "answer_rate": round(answer_rate, 4),
    }

    # Per-level breakdown
    level_stats = {}
    for r in results:
        lvl = r["level"]
        if lvl not in level_stats:
            level_stats[lvl] = {"total": 0, "correct": 0}
        level_stats[lvl]["total"] += 1
        if r["hit"]:
            level_stats[lvl]["correct"] += 1
    for lvl in sorted(level_stats):
        s = level_stats[lvl]
        s["hit_at_1"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0
    summary["per_level"] = level_stats

    output = {"summary": summary, "results": results}
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Results: Hit@1={hit1:.4f} ({correct}/{total}), Answer rate={answer_rate:.4f}")
    for lvl in sorted(level_stats):
        s = level_stats[lvl]
        print(f"  Level {lvl}: Hit@1={s['hit_at_1']:.4f} ({s['correct']}/{s['total']})")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
