# DeepRetrieval Faithful Reproduction

Faithful reproduction of DeepRetrieval (arXiv:2503.00223) on our ecommerce dataset.

## Key Principles
- Uses the author's original code from https://github.com/pat-jj/DeepRetrieval
- PPO algorithm (not GRPO) with GAE advantage estimation — matches the paper
- Same base model (Qwen3-4B) as our method for fair comparison
- Only data-specific files are added; core training logic is untouched

## Files in this directory
These files should be placed into the DeepRetrieval repo at the specified paths:

| File | Target path in DeepRetrieval repo |
|------|----------------------------------|
| `1_convert_format.py` | `code/src/Lucene/ecommerce/1_convert_format.py` |
| `2_build_database.sh` | `code/src/Lucene/ecommerce/2_build_database.sh` |
| `search.py` | `code/src/Lucene/ecommerce/search.py` |
| `data_preprocess_ecommerce.py` | `code/data_preprocess/ecommerce.py` |
| `reward_ecommerce.py` | `code/verl/utils/reward_score/ecommerce.py` |
| `train_ecommerce.sh` | `code/scripts/train/ecommerce.sh` |
| `setup_102.sh` | Setup script for server 102 |

## Minimal changes to author's code
1. `code/verl/trainer/main_ppo.py`: Add `elif 'ecommerce' in data_source:` in `_select_rm_score_fn`
2. `code/requirements.txt`: Relax `transformers` and `vllm` version for Qwen3-4B support
