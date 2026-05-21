"""
Reward function for Search-R1 RL training.
"""
import re
import json
import string
import random


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def parse_ground_truth(ground_truth):
    """Normalize ground_truth to a list of target strings.
    
    verl passes the reward_model column which has structure:
    {'style': 'rule', 'ground_truth': {'target': [...], 'db_name': '...'}}
    """
    if isinstance(ground_truth, dict):
        if 'ground_truth' in ground_truth:
            inner = ground_truth['ground_truth']
            if isinstance(inner, dict):
                target = inner.get('target', [])
                if hasattr(target, 'tolist'):
                    return target.tolist()
                return list(target) if not isinstance(target, list) else target
            if isinstance(inner, list):
                return inner
            return [str(inner)]
        target = ground_truth.get('target', [])
        if hasattr(target, 'tolist'):
            return target.tolist()
        return list(target) if not isinstance(target, list) else target
    if isinstance(ground_truth, list):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            return parse_ground_truth(parsed)
        except (json.JSONDecodeError, TypeError):
            return [ground_truth]
    return []


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    target_ids = parse_ground_truth(ground_truth)

    if not target_ids:
        return 0.0

    answer = extract_solution(solution_str)
    do_print = random.randint(1, 32) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers (first 5): {target_ids[:5]}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string (last 800): {solution_str[-800:]}")

    if answer is None:
        return 0.0

    if em_check(answer, target_ids):
        if do_print:
            print(f">> CORRECT!")
        return 1.0

    return 0.0
