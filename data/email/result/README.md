# Email Evaluation Results

All results evaluated on `queries_email_test_step9.jsonl` (N=3000, unless noted).

## Directory Structure

### `b1_pure_retrieval/` — B1: Pure Retrieval Baselines
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| b1_bm25.json | BM25 | 3000 | 0.731 |
| b1_colqwen.json | ColQwen (Qwen3-Embedding-8B) | 3000 | 0.472 |
| b1_rrf.json | RRF (BM25+ColQwen, k=60) | 3000 | 0.687 |
| b1_bgem3.json | BGE-M3 | 3000 | 0.599 |
| b1_reranker.json | BGE-Reranker-v2-M3 on RRF | 3000 | 0.836 |

### `b2_llm_enhanced/` — B2: LLM-Augmented Retrieval
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| b2_hyde.json | HyDE (GPT-5.4) | 300 | 0.523 |
| b2_ircot.json | IRCoT 2-round (GPT-5.4) | 3000 | 0.511 |
| b2_rankgpt.json | RankGPT listwise rerank (GPT-5.4) | 3000 | 0.888 |

### `b3_constrained_structured/` — B3: Constrained Structured Query
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| b3_langchain_sqr.json | LangChain SQR (GPT-5.4) | 300 | 0.617 |

### `ours_rl/` — Our Method: RL-trained DSL (Qwen3-4B)
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| predictions_dapo_step843.json | DAPO 4B step843 (best) | 3000 | 0.909 |
| predictions_grpo_step843.json | GRPO 4B step843 | 3000 | 0.906 |

### `ours_sft/` — Our Method: SFT baselines
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| predictions_sft_email.json | SFT 4B | 3000 | 0.891 |
| predictions_base_email.json | Base 4B (zero-shot) | 3000 | 0.243 |

### `commercial_llm/` — Commercial LLM DSL generation
| File | Method | N | Hit@1 |
|------|--------|---|-------|
| predictions_gpt54.json | GPT-5.4 | 2999 | 0.854 |
| predictions_gpt55.json | GPT-5.5 | 3000 | 0.855 |
| predictions_opus47.json | Opus 4.7 | 3000 | 0.830 |

## Notes
- Hit@1 used as primary metric (equivalent to NDCG@1 for single ground truth)
- B1-B3 are ranked-list baselines (direct Hit@1 on ranked results)
- `eval_*.json` files contain detailed metrics (Hit@1, Hit@3, MRR, per-level breakdown)
- Root-level files are kept for backward compatibility; subdirectories contain organized copies
- B4 (LightRAG) is running — results pending
