"""
Patch for DeepRetrieval's code/verl/trainer/main_ppo.py

Add these lines in _select_rm_score_fn(), before the final 'else: raise NotImplementedError':

    elif 'ecommerce' in data_source:
        from verl.utils.reward_score import ecommerce
        return ecommerce.compute_score

And in the RewardManager.__call__() method, add 'ecommerce' to the condition
that passes data_source to compute_score_fn:

    elif 'scifact' in data_source or 'fiqa' in data_source or ... or 'ecommerce' in data_source:
        score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, data_source=data_source)

This is the complete diff:
"""

# --- To apply automatically, run: ---
# python main_ppo_patch.py

import re

MAIN_PPO_PATH = "verl/trainer/main_ppo.py"

def patch():
    with open(MAIN_PPO_PATH) as f:
        content = f.read()

    if 'ecommerce' in content:
        print("Already patched!")
        return

    # 1. Add to _select_rm_score_fn
    old = "    else:\n        raise NotImplementedError"
    new = (
        "    elif 'ecommerce' in data_source:\n"
        "        from verl.utils.reward_score import ecommerce\n"
        "        return ecommerce.compute_score\n"
        "    else:\n"
        "        raise NotImplementedError"
    )
    content = content.replace(old, new)

    # 2. Add to RewardManager.__call__ dispatch
    old2 = "or 'msmarco' in data_source:"
    new2 = "or 'msmarco' in data_source or 'ecommerce' in data_source:"
    content = content.replace(old2, new2)

    with open(MAIN_PPO_PATH, 'w') as f:
        f.write(content)

    print("Patched main_ppo.py successfully")


if __name__ == '__main__':
    patch()
