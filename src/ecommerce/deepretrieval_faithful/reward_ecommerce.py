#!/usr/bin/env python3
"""Reward function for DeepRetrieval faithful reproduction on ecommerce.

Follows the exact reward logic from DeepRetrieval's verl/utils/reward_score/msmarco_beir.py:
- Format reward: +0.1 for correct <think>...</think><answer>{"query":"..."}</answer>, -2 otherwise
- Answer reward: NDCG@k from BM25 retrieval (k=1000 for train, k=10 for val/test)
- Total: format_score + answer_score if answer_score > 0, else 0 (good format) or format_score (bad format)

Adaptations for verl 0.7.1:
- extra_info parameter (verl 0.7.1 passes this)
- Response-only input handling (verl 0.7.1 sends response only, not full prompt+response)
- Multi-target NDCG (our data has multiple relevant products per query)
- LuceneSearcher instead of batch_search (functionally equivalent for single queries)
"""
import json
import os
import random
import re

import numpy as np
from pyserini.search.lucene import LuceneSearcher

INDEX_DIR = os.environ.get(
    "LUCENE_INDEX_DIR",
    os.path.expanduser("~/LocalSearch/deepretrieval/index")
)

_searcher = None


def get_searcher():
    global _searcher
    if _searcher is None:
        print(f"Loading Lucene index from {INDEX_DIR}")
        _searcher = LuceneSearcher(index_dir=INDEX_DIR)
        print(f"Index loaded: {_searcher.num_docs} documents")
    return _searcher


def dcg_at_k(retrieved, target_set, k):
    retrieved = retrieved[:k]
    gains = np.array([1.0 if doc in target_set else 0.0 for doc in retrieved])
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(retrieved, targets, k):
    target_set = set(targets) if isinstance(targets, list) else {targets}
    dcg = dcg_at_k(retrieved, target_set, k)
    ideal_retrieved = list(target_set)[:k]
    ideal_dcg = dcg_at_k(ideal_retrieved, target_set, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def extract_solution(solution_str):
    """Extract answer from model output.

    Author's version expects full prompt+response with "Assistant:" header.
    verl 0.7.1 passes response-only, so we also handle that case.
    """
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        processed_str = solution_str

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)
    if matches:
        return matches[-1].strip(), processed_str
    return None, processed_str


def validate_response_structure(processed_str, do_print=False):
    """Validate <think>...</think><answer>...</answer> structure.

    Author requires exactly 1 of each tag in order.
    For response-only input (verl 0.7.1), <think> may be absent since it's in the prompt.
    """
    think_start_count = processed_str.count("<think>")
    think_end_count = processed_str.count("</think>")
    answer_start_count = processed_str.count("<answer>")
    answer_end_count = processed_str.count("</answer>")

    if think_start_count == 1 and think_end_count == 1 and answer_start_count == 1 and answer_end_count == 1:
        p_ts = processed_str.find("<think>")
        p_te = processed_str.find("</think>")
        p_as = processed_str.find("<answer>")
        p_ae = processed_str.find("</answer>")
        return p_ts < p_te < p_as < p_ae

    if think_start_count == 0 and think_end_count == 1 and answer_start_count == 1 and answer_end_count == 1:
        p_te = processed_str.find("</think>")
        p_as = processed_str.find("<answer>")
        p_ae = processed_str.find("</answer>")
        return p_te < p_as < p_ae

    return False


def check_json_format(json_str, do_print=False):
    try:
        if not json_str:
            return False
        data = json.loads(json_str)
        return "query" in data
    except json.JSONDecodeError:
        return False


def retrieve_products(query, top_k=1000):
    searcher = get_searcher()
    hits = searcher.search(query, k=top_k)
    return [hit.docid for hit in hits]


def compute_score(solution_str, ground_truth, data_source, extra_info=None,
                  format_reward=0.1, answer_reward=1.0):
    """Compute reward score — identical logic to DeepRetrieval msmarco_beir.py."""
    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]
    label = [str(x) for x in targets]

    answer_text, processed_str = extract_solution(solution_str)
    do_print = random.randint(1, 32) == 1

    response_format_ok = validate_response_structure(processed_str, do_print)
    json_format_ok = check_json_format(answer_text, do_print)
    format_correct = response_format_ok and json_format_ok

    format_score = format_reward if format_correct else -2

    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str[:300]}")
        print(f"Target: {label[:3]}... |")

    answer_score = 0
    if 'test' in data_source or 'val' in data_source:
        top_k = 10
    else:
        top_k = 1000

    if format_correct and answer_text:
        try:
            data = json.loads(answer_text)
            query = data["query"]
            results = retrieve_products(query, top_k=top_k)
            answer_score = ndcg_at_k(results, label, top_k)
        except Exception as e:
            if do_print:
                print(f"[Error] Retrieval failed: {e}")
            answer_score = -2

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score

    if do_print:
        print(f" Final Score: Format={format_score}, Answer={answer_score:.4f}, Total={total_score:.4f}")
        print("=" * 60)

    return total_score
