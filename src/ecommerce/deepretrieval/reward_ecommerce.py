#!/usr/bin/env python3
"""Reward function for DeepRetrieval on ecommerce product retrieval.

Adapted from DeepRetrieval's verl/utils/reward_score/msmarco.py.
Uses Pyserini LuceneSearcher for BM25 retrieval and NDCG@k as reward.
"""
import json
import os
import random
import re

import numpy as np
from pyserini.search.lucene import LuceneSearcher

INDEX_DIR = "/home/sunzhen/DeepRetrieval_ecommerce/indexes/lucene-index-products-train"

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
    target_set = set(targets)
    dcg = dcg_at_k(retrieved, target_set, k)
    ideal_retrieved = list(target_set)[:k]
    ideal_dcg = dcg_at_k(ideal_retrieved, target_set, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        processed_str = solution_str

    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)
    if matches:
        return matches[-1].strip(), processed_str
    return None, processed_str


def validate_response_structure(processed_str):
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


def check_json_format(json_str):
    try:
        if not json_str:
            return False
        data = json.loads(json_str)
        return "query" in data
    except json.JSONDecodeError:
        return False


def retrieve_products(query, top_k=100):
    searcher = get_searcher()
    hits = searcher.search(query, k=top_k)
    return [hit.docid for hit in hits]


def compute_score(solution_str, ground_truth, data_source, extra_info=None, format_reward=0.1, answer_reward=1.0):
    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]

    answer_text, processed_str = extract_solution(solution_str)
    do_print = random.randint(1, 16) == 1

    response_format_ok = validate_response_structure(processed_str)
    json_format_ok = check_json_format(answer_text)
    format_correct = response_format_ok and json_format_ok

    format_score = format_reward if format_correct else -2

    answer_score = 0
    top_k = 1000

    if format_correct and answer_text:
        try:
            data = json.loads(answer_text)
            query = data["query"]
            results = retrieve_products(query, top_k=top_k)
            answer_score = ndcg_at_k(results, targets, top_k)
        except Exception as e:
            if do_print:
                print(f"[Error] Retrieval failed: {e}")
            answer_score = -2

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        total_score = 0 if format_score > 0 else format_score

    if do_print:
        print(f"{'='*60}")
        print(f"Solution: {solution_str[:300]}...")
        print(f"Format: {format_score}, Answer: {answer_score:.4f}, Total: {total_score:.4f}")
        print(f"{'='*60}")

    return total_score
