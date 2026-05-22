"""Patch qa_em.py to handle list ground_truth format."""
import sys

path = "/root/autodl-tmp/Search-R1/verl/utils/reward_score/qa_em.py"
with open(path, "r") as f:
    content = f.read()

# Replace all ground_truth['target'] with a helper function call
helper = '''
def _get_targets(ground_truth):
    if isinstance(ground_truth, list):
        return ground_truth
    if isinstance(ground_truth, dict):
        return ground_truth.get('target', [])
    return [str(ground_truth)]

'''

# Add helper after imports
content = content.replace("import random\n", "import random\n" + helper)

# Replace ground_truth['target'] with _get_targets(ground_truth)
content = content.replace("ground_truth['target']", "_get_targets(ground_truth)")

with open(path, "w") as f:
    f.write(content)

print("Patched qa_em.py successfully")
