"""
Search-R1 evaluation: vLLM batch inference + online BM25 multi-turn search.
Matches verl's search_r1 format for result formatting.
Usage: python eval_search_r1_vllm.py --model_path <ckpt> --test_parquet <path> --output <path>
"""
import json, re, string, argparse, requests
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for ga in golden_answers:
        if normalize_answer(ga) == normalized_prediction:
            return True
    return False


def extract_answer(text):
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_search_query(text):
    matches = list(re.finditer(r'<search>(.*?)</search>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def do_bm25_search(query, topk=3, url="http://127.0.0.1:8000/retrieve"):
    """Call BM25 server with return_scores=True to match verl training format."""
    try:
        resp = requests.post(url, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=10)
        data = resp.json()
        if isinstance(data, dict) and 'result' in data:
            results = data['result']
            if isinstance(results, list) and len(results) > 0:
                return results[0]
        elif isinstance(data, list) and len(data) > 0:
            return data[0]
        return []
    except:
        return []


def format_search_results(results):
    """Format results matching verl's _passages2string format."""
    parts = []
    for idx, doc in enumerate(results):
        if isinstance(doc, dict):
            if "document" in doc:
                contents = doc["document"].get("contents", "")
            else:
                contents = doc.get("contents", doc.get("text", str(doc)))

            lines = contents.split("\n")
            title = lines[0] if lines else ""
            text = "\n".join(lines[1:]) if len(lines) > 1 else ""
            parts.append(f"Doc {idx + 1} (Title: {title})\n{text}")
        else:
            parts.append(f"Doc {idx + 1}\n{str(doc)}")
    return "\n\n".join(parts)


def build_chat_text(tokenizer_name, messages):
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--output", default="/root/autodl-tmp/search_r1/eval_results/search_r1_step800_results.json")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retrieval_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        trust_remote_code=True,
    )
    search_stop_params = SamplingParams(temperature=0, max_tokens=500, stop=["<|im_end|>", "</search>"])

    print(f"Loading test data: {args.test_parquet}")
    df = pd.read_parquet(args.test_parquet)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    results = []
    correct = 0
    total = 0
    no_answer = 0

    for idx, row in df.iterrows():
        prompt = row['prompt']
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

        rm = row['reward_model']
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = rm.get('ground_truth', {})
        if isinstance(gt, dict):
            targets = gt.get('target', [])
        elif isinstance(gt, list):
            targets = gt
        else:
            targets = []
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()

        messages = list(prompt)
        final_answer = None
        all_responses = []

        for turn in range(args.max_turns):
            text = build_chat_text("qwen", messages)
            outputs = llm.generate([text], search_stop_params)
            response = outputs[0].outputs[0].text
            stop_reason = outputs[0].outputs[0].stop_reason

            if stop_reason == "</search>":
                response = response + "</search>"

            all_responses.append(response)
            answer = extract_answer(response)
            if answer:
                final_answer = answer
                break

            search_q = extract_search_query(response)
            if not search_q:
                break

            search_results = do_bm25_search(search_q, topk=args.topk, url=args.retrieval_url)
            info_text = format_search_results(search_results)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"<information>\n{info_text}\n</information>"})

        is_correct = False
        if final_answer and targets:
            is_correct = em_check(final_answer, targets)

        if final_answer is None:
            no_answer += 1

        if targets:
            total += 1
            if is_correct:
                correct += 1

        results.append({
            "idx": idx,
            "answer": final_answer,
            "correct": is_correct,
            "targets": targets[:3] if isinstance(targets, list) else [],
            "n_turns": len(all_responses),
            "has_gt": len(targets) > 0 if isinstance(targets, list) else False,
        })

        if (idx + 1) % 100 == 0:
            rate = correct / total if total > 0 else 0
            print(f"[{idx+1}/{len(df)}] EM={correct}/{total}={rate:.4f} no_answer={no_answer}")

    em_score = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"FINAL: EM={correct}/{total}={em_score:.4f}")
    print(f"No answer: {no_answer}/{len(df)}")
    print(f"Empty GT skipped: {len(df) - total}")

    with open(args.output, 'w') as f:
        json.dump({
            "em_score": em_score,
            "correct": correct,
            "total": total,
            "no_answer": no_answer,
            "empty_gt": len(df) - total,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
