# ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis

<p align="center">
  <img src="figures/fig_method_overview.png" width="90%" alt="ProRetrieval Framework Overview">
</p>

## Overview

**ProRetrieval** recasts the language model as a *retrieval orchestrator*: given a natural-language query, it synthesizes an executable program in a hybrid DSL that interleaves SQL operators over structured fields with vector-retrieval primitives over text and images. SQL's logical algebra provides the orchestration skeleton, while vector retrieval is incorporated as a first-class primitive through candidate-set placeholders.

We train Qwen3-4B with GRPO and DAPO under a hierarchical four-term reward on two new benchmarks (Amazon e-commerce and Enron email). Our 4B model surpasses GPT-5.5 and a comprehensive suite of retrieval, reranking, structured-query, and graph-based baselines.

### Key Results

| Method | E-commerce Hit@1 | Email Hit@1 |
|--------|:---:|:---:|
| BM25 | 0.571 | 0.731 |
| BGE-Reranker | 0.618 | 0.836 |
| RankGPT (GPT-5.4) | 0.768 | 0.888 |
| DeepRetrieval (RL) | 0.625 | 0.745 |
| GPT-5.5 (few-shot) | 0.693 | 0.855 |
| **ProRetrieval 4B DAPO** | **0.808** | **0.909** |
| **ProRetrieval 8B SFT** | **0.820** | 0.820 |

## Project Structure

```
├── src/                        # Core source code
│   ├── ecommerce/              # E-commerce domain modules
│   ├── email/                  # Email domain modules
│   ├── eval_ecommerce.py       # E-commerce evaluation
│   ├── eval_email.py           # Email evaluation
│   ├── inference_ecommerce.py  # E-commerce inference
│   ├── inference_email.py      # Email inference
│   └── inference_closedapi.py  # Commercial LLM inference
├── data/                       # Benchmark datasets
│   ├── ecommerce/              # Amazon ESCI + Reviews
│   └── email/                  # Enron email corpus
├── sft_4b/                     # SFT training scripts
│   ├── train_sft.py            # SFT training script
│   ├── prepare_data.py         # Data preparation
│   ├── run_sft.sh              # SFT launch script
│   └── ds_zero2.json           # DeepSpeed ZeRO-2 config
├── rl_ecommerce/               # RL training (e-commerce)
│   ├── reward_dsl.py           # DSL reward function
│   ├── run_rl_grpo.sh          # GRPO training script
│   ├── run_rl_verl.sh          # verl-based RL script
│   └── embedding_server.py     # Vector retrieval server
├── rl_8b/                      # RL training (8B model)
├── search_r1/                  # Search-R1 baseline
├── scripts/                    # Analysis & evaluation scripts
├── tools/                      # Data preprocessing utilities
├── results/                    # Experiment results
└── figures/                    # Paper figures
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/zjw49246/LocalSearch-R1.git
cd LocalSearch-R1

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Preprocess e-commerce data
python tools/preprocess_product.py

# Preprocess email data
python tools/preprocess_email.py

# Prepare SFT training data
python sft_4b/prepare_data.py
```

### 3. SFT Training

```bash
# Train Qwen3-4B with SFT
bash sft_4b/run_sft.sh
```

### 4. RL Training (GRPO/DAPO)

```bash
# Start the embedding server for vector retrieval
python rl_ecommerce/embedding_server.py &

# Run GRPO training on e-commerce
bash rl_ecommerce/run_rl_grpo.sh

# Or run with verl framework
bash rl_ecommerce/run_rl_verl.sh
```

### 5. Inference & Evaluation

```bash
# Run inference on e-commerce
python src/inference_ecommerce.py --model_path <checkpoint_path>

# Evaluate
python src/eval_ecommerce.py --results_path <results_file>

# Run inference on email
python src/inference_email.py --model_path <checkpoint_path>
python src/eval_email.py --results_path <results_file>

# Commercial LLM inference (GPT-5.5, Opus 4.7, etc.)
python src/inference_closedapi.py --model gpt-5.5 --domain ecommerce
```

## Hybrid Retrieval DSL

A program in our DSL is a JSON object with two fields:

```json
{
  "sql": "SELECT DISTINCT id FROM products WHERE brand = 'Sony' AND price < 200 AND product_id IN <text_0> AND product_id IN <image_0>",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "strong noise cancelling"},
    {"type": "image", "field": "image_url", "query": "matte black wireless headphone"}
  ]
}
```

**Execution pipeline:**
1. Each entry in `retrieval_list` is dispatched to the corresponding encoder (text or image) for top-K nearest-neighbor search
2. Placeholders (`<text_0>`, `<image_0>`) are substituted with returned candidate IDs
3. The resulting SQL is executed against the relational backend for logical fusion

## Hierarchical Composite Reward

The reward function mirrors layered program correctness:

$$R = S_f + S_e + S_r + S_l$$

| Term | Range | Description |
|------|-------|-------------|
| $S_f$ | {0, 1} | Format validity (parses as valid DSL JSON) |
| $S_e$ | {0, 1} | Execution success (SQL compiles, placeholders resolve) |
| $S_r$ | [0, 1] | Retrieval accuracy (rank-1 hit) |
| $S_l$ | [0, 1] | Length budget (penalizes verbose programs) |

## Benchmarks

| Domain | Corpus Size | Modalities | Train Queries | Test Queries |
|--------|:-----------:|:----------:|:-------------:|:------------:|
| E-commerce | 3,000 products | Structured + Text + Image | 20,000 | 3,000 |
| Email | 5,000 emails | Structured + Text | 20,000 | 3,000 |

Queries are stratified into three complexity levels:
- **L1** (~40%): Single condition
- **L2** (~40%): 2-3 conditions with AND/OR
- **L3** (~20%): 4-5 conditions with OR/NOT/nesting

## Citation

```bibtex
@article{proretrieval2026,
  title={ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis},
  author={Chengsong You and Zhen Sun and Yunhai Hu and Junwei Zhou and Xiaoyu Cao and Binyu Li and Ziyan Zhao and Weiyao Wang and Liren Lu and Zhijie Ye and Yumo Cao and Yitao Long and Yiwei Xu and Qiyi Jiang and Xuanyi Fu and Yufan Chen and Yilun Li and Rongkang Xiong and Yiran Zou and Nan Du},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
