# Ecommerce Evaluation Results

All results evaluated on `queries_product_test_step9_gt_v2.jsonl` (N=3000, unless noted).

## Directory Structure

### `b1_pure_retrieval/` — B1: Pure Retrieval Baselines
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| b1_bm25.json | BM25 | 3000 | 0.571 |
| b1_colqwen.json | ColQwen (Qwen3-Embedding-8B) | 3000 | 0.518 |
| b1_rrf.json | RRF (BM25+ColQwen, k=60) | 3000 | 0.576 |
| b1_bgem3.json | BGE-M3 | 3000 | 0.571 |
| b1_reranker.json | BGE-Reranker-v2-M3 on RRF | 3000 | 0.618 |
| b1_dense_image.json | Dense Image (Qwen3-VL) | 3000 | 0.058 |

### `b2_llm_enhanced/` — B2: LLM-Augmented Retrieval
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| b2_hyde.json | HyDE (GPT-5.4) | 3000 | 0.549 |
| b2_ircot.json | IRCoT 2-round (GPT-5.4) | 3000 | 0.565 |
| b2_rankgpt.json | RankGPT listwise rerank (GPT-5.4) | 3000 | 0.768 |

### `b3_constrained_structured/` — B3: Constrained Structured Query
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| b3_deepretrieval.json | DeepRetrieval (Boolean+BM25, N=2737) | 2737 | 0.625 |
| b3_deepretrieval_full.json | DeepRetrieval full eval | 2737 | 0.625 |
| b3_langchain_sqr.json | LangChain SQR (GPT-5.4) | 3000 | 0.606 |
| b3_sql_r1.json | SQL-R1 RL (4B) | 3000 | 0.620 |
| b3_search_r1_step800.json | Search-R1 (BM25 multi-turn RL) | 3000 | 0.407 |

### `b4_knowledge_graph/` — B4: Knowledge Graph Retrieval
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| b4_graphrag.json | GraphRAG (entity matching, GPT-5.4) | 3000 | 0.500 |
| b4_lightrag.json | LightRAG (EMNLP 2025, hybrid KG+vec) | 3000 | 0.588 |
| eval_b4_graphrag.json | GraphRAG eval metrics | — | — |
| eval_b4_lightrag.json | LightRAG eval metrics | — | — |

### `b5_fewshot_dsl/` — B5: Few-shot DSL (ICL, no training)
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| predictions_fewshot_gpt55.json | GPT-5.5 few-shot DSL | 300 | 0.548 |
| predictions_fewshot_opus47.json | Opus 4.7 few-shot DSL | 300 | 0.439 |

### `ours_rl/` — Our Method: RL-trained DSL (Qwen3-4B/8B)
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| predictions_grpo4b_step200.json | GRPO 4B step200 | 3000 | 0.872 |
| predictions_grpo4b_step300.json | GRPO 4B step300 (best) | 3000 | 0.893 |
| predictions_grpo4b_step550.json | GRPO 4B step550 | 3000 | 0.869 |
| predictions_grpo4b_step1200.json | GRPO 4B step1200 (collapsed) | 3000 | 0.003 |
| predictions_dapo4b_step420.json | DAPO 4B step420 | 3000 | 0.876 |
| predictions_grpo8b_step200.json | GRPO 8B step200 | 3000 | 0.819 |
| predictions_grpo8b_step300.json | GRPO 8B step300 | 3000 | 0.820 |
| predictions_grpo8b_step400.json | GRPO 8B step400 | 3000 | 0.820 |

### `ours_sft/` — Our Method: SFT baselines
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| predictions_sft_4b.json | SFT 4B | 3000 | 0.787 |
| predictions_sft_1.7b.json | SFT 1.7B | 3000 | — |
| predictions_sft8b_2.0ep.json | SFT 8B (2.0ep) | 3000 | 0.820 |
| predictions_base4b.json | Base 4B (zero-shot) | 3000 | — |
| predictions_base8b.json | Base 8B (zero-shot) | 3000 | — |

### `commercial_llm/` — Commercial LLM DSL generation
| File | Method | N | NDCG@1 |
|------|--------|---|--------|
| predictions_gpt54_azure.json | GPT-5.4 | 3000 | 0.654 |
| predictions_gpt54_thinking.json | GPT-5.4 thinking (300) | 300 | — |
| predictions_gpt55.json | GPT-5.5 | 3000 | 0.748 |
| predictions_opus47.json | Opus 4.7 | 3000 | 0.727 |
| predictions_opus46.json | Opus 4.6 | 3000 | — |
| predictions_gemini_31pro_v2.json | Gemini 3.1 Pro | 3000 | — |

## Notes
- NDCG@1 values are "optimistic" for DSL methods (exec failures scored as 0)
- B1-B4 are ranked-list baselines (direct NDCG on ranked results)
- `eval_*.json` files contain detailed metrics (Hit@1, Hit@3, MRR, per-level breakdown)
- Root-level files are kept for backward compatibility; subdirectories contain organized copies
