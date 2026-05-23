#!/usr/bin/env python3
"""Run all statistics-only analysis experiments (E1, E3, E5, E6, E7, E8, E9).
Outputs go to results/analysis/
"""
import json, os, math, glob, re
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECOM_GT = f"{BASE}/data/ecommerce/test/queries_product_test_step9_gt_v2.jsonl"
EMAIL_GT = f"{BASE}/data/email/test/queries_email_test_step9.jsonl"
ECOM_RESULT = f"{BASE}/data/ecommerce/result"
EMAIL_RESULT = f"{BASE}/data/email/result"
OUT_DIR = f"{BASE}/results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def load_gt(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def load_pred(path):
    return json.load(open(path))

def compute_metrics(predictions, queries, top_k=20):
    hit1 = hit3 = mrr = n = 0
    for q in queries:
        qid = q["query_id"]
        if qid not in predictions:
            continue
        gt = set(q.get("gt_document_ids") or q.get("gold_ids") or [])
        if not gt:
            continue
        ranked = predictions.get(qid, [])[:top_k]
        n += 1
        if ranked and ranked[0] in gt:
            hit1 += 1
        if any(r in gt for r in ranked[:3]):
            hit3 += 1
        for j, r in enumerate(ranked):
            if r in gt:
                mrr += 1.0 / (j + 1)
                break
    if n == 0:
        return {}
    return {"n": n, "hit1": round(hit1/n, 4), "hit3": round(hit3/n, 4), "mrr": round(mrr/n, 4)}

def compute_by_group(predictions, queries, group_key, top_k=20):
    groups = defaultdict(lambda: {"hit1":0,"hit3":0,"mrr":0.0,"n":0})
    for q in queries:
        qid = q["query_id"]
        if qid not in predictions:
            continue
        gt = set(q.get("gt_document_ids") or q.get("gold_ids") or [])
        if not gt:
            continue
        if group_key == "level":
            gk = str(q.get("level", "?"))
        elif group_key == "modality":
            gk = "+".join(sorted(q.get("modalities", [])))
        else:
            gk = "all"
        ranked = predictions.get(qid, [])[:top_k]
        groups[gk]["n"] += 1
        if ranked and ranked[0] in gt:
            groups[gk]["hit1"] += 1
        if any(r in gt for r in ranked[:3]):
            groups[gk]["hit3"] += 1
        for j, r in enumerate(ranked):
            if r in gt:
                groups[gk]["mrr"] += 1.0 / (j + 1)
                break
    result = {}
    for gk in sorted(groups.keys()):
        d = groups[gk]
        n = d["n"]
        result[gk] = {"n": n, "hit1": round(d["hit1"]/n, 4), "hit3": round(d["hit3"]/n, 4), "mrr": round(d["mrr"]/n, 4)}
    return result

def get_dsl_field(eval_path, field):
    if not os.path.exists(eval_path):
        return None
    d = json.load(open(eval_path))
    if field not in d:
        return None
    if field == "overall":
        o = d["overall"]
        return {"n": o.get("count") or o.get("total", 0),
                "hit1": round(o.get("hit_at_1", 0), 4),
                "hit3": round(o.get("hit_at_3", 0), 4),
                "mrr": round(o.get("mrr", 0), 4),
                "exec_rate": o.get("exec_success_rate", None)}
    elif field == "by_level":
        result = {}
        for lv, vals in d["by_level"].items():
            result[lv] = {"n": vals.get("count", 0),
                          "hit1": round(vals.get("hit_at_1", 0), 4),
                          "hit3": round(vals.get("hit_at_3", 0), 4),
                          "mrr": round(vals.get("mrr", 0), 4)}
        return result
    elif field == "by_modality":
        result = {}
        for mod, vals in d["by_modality"].items():
            result[mod] = {"n": vals.get("count", 0),
                           "hit1": round(vals.get("hit_at_1", 0), 4),
                           "hit3": round(vals.get("hit_at_3", 0), 4),
                           "mrr": round(vals.get("mrr", 0), 4)}
        return result
    return None

# ============================================================
# Method definitions
# ============================================================

ECOM_METHODS = {
    "B1_BM25": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_bm25.json"},
    "B1_ColQwen": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_colqwen.json"},
    "B1_RRF": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_rrf.json"},
    "B1_BGE-M3": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_bgem3.json"},
    "B1_Reranker": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_reranker.json"},
    "B1_DenseImage": {"pred": f"{ECOM_RESULT}/b1_pure_retrieval/b1_dense_image.json"},
    "B2_HyDE": {"pred": f"{ECOM_RESULT}/b2_llm_enhanced/b2_hyde.json"},
    "B2_IRCoT": {"pred": f"{ECOM_RESULT}/b2_llm_enhanced/b2_ircot.json"},
    "B2_RankGPT": {"pred": f"{ECOM_RESULT}/b2_llm_enhanced/b2_rankgpt.json"},
    "B3_DeepRetrieval": {"pred": f"{ECOM_RESULT}/b3_constrained_structured/b3_deepretrieval.json"},
    "B3_SQR": {"pred": f"{ECOM_RESULT}/b3_constrained_structured/b3_langchain_sqr.json"},
    "B3_SQL-R1": {"pred": f"{ECOM_RESULT}/b3_constrained_structured/b3_sql_r1.json"},
    "B3_Search-R1": {"pred": f"{ECOM_RESULT}/b3_constrained_structured/b3_search_r1_step800.json"},
    "B4_GraphRAG": {"pred": f"{ECOM_RESULT}/b4_knowledge_graph/b4_graphrag.json"},
    "B4_LightRAG": {"pred": f"{ECOM_RESULT}/b4_knowledge_graph/b4_lightrag.json"},
    "B5_GPT55_fewshot": {"eval": f"{ECOM_RESULT}/b5_fewshot_dsl/eval_predictions_fewshot_gpt55.json"},
    "B5_Opus47_fewshot": {"eval": f"{ECOM_RESULT}/b5_fewshot_dsl/eval_predictions_fewshot_opus47.json"},
    "GPT-5.4": {"eval": f"{ECOM_RESULT}/commercial_llm/eval_gpt54.json"},
    "GPT-5.5": {"eval": f"{ECOM_RESULT}/commercial_llm/eval_gpt55.json"},
    "Opus_4.7": {"eval": f"{ECOM_RESULT}/commercial_llm/eval_opus47.json"},
    "Opus_4.6": {"eval": f"{ECOM_RESULT}/commercial_llm/eval_opus46.json"},
    "Gemini_3.1Pro": {"eval": f"{ECOM_RESULT}/commercial_llm/eval_gemini31pro.json"},
    "SFT_1.7B": {"eval": f"{ECOM_RESULT}/ours_sft/eval_sft_1.7b.json"},
    "SFT_4B": {"eval": f"{ECOM_RESULT}/ours_sft/eval_sft4b.json"},
    "SFT_8B": {"eval": f"{ECOM_RESULT}/ours_sft/eval_sft8b.json"},
    "Base_4B": {"eval": f"{ECOM_RESULT}/ours_sft/eval_base4b.json"},
    "Base_8B": {"eval": f"{ECOM_RESULT}/ours_sft/eval_base8b.json"},
    "GRPO_4B_step200": {"eval": f"{ECOM_RESULT}/ours_rl/eval_grpo4b_step200.json"},
    "GRPO_4B_step300": {"eval": f"{ECOM_RESULT}/ours_rl/eval_grpo4b_step300.json"},
    "GRPO_4B_step550": {"eval": f"{ECOM_RESULT}/ours_rl/eval_grpo4b_step550.json"},
    "DAPO_4B_step420": {"eval": f"{ECOM_RESULT}/ours_rl/eval_dapo4b_step420.json"},
    "GRPO_8B_step150": {"eval": f"{ECOM_RESULT}/ours_rl/eval_grpo8b_step150.json"},
}

EMAIL_METHODS = {
    "B1_BM25": {"pred": f"{EMAIL_RESULT}/b1_pure_retrieval/b1_bm25.json"},
    "B1_ColQwen": {"pred": f"{EMAIL_RESULT}/b1_pure_retrieval/b1_colqwen.json"},
    "B1_RRF": {"pred": f"{EMAIL_RESULT}/b1_pure_retrieval/b1_rrf.json"},
    "B1_BGE-M3": {"pred": f"{EMAIL_RESULT}/b1_pure_retrieval/b1_bgem3.json"},
    "B1_Reranker": {"pred": f"{EMAIL_RESULT}/b1_pure_retrieval/b1_reranker.json"},
    "B2_HyDE": {"pred": f"{EMAIL_RESULT}/b2_llm_enhanced/b2_hyde.json"},
    "B2_IRCoT": {"pred": f"{EMAIL_RESULT}/b2_llm_enhanced/b2_ircot.json"},
    "B2_RankGPT": {"pred": f"{EMAIL_RESULT}/b2_llm_enhanced/b2_rankgpt.json"},
    "B3_SQR": {"pred": f"{EMAIL_RESULT}/b3_constrained_structured/b3_langchain_sqr.json"},
    "GPT-5.4": {"eval": f"{EMAIL_RESULT}/commercial_llm/eval_gpt54.json"},
    "GPT-5.5": {"eval": f"{EMAIL_RESULT}/commercial_llm/eval_gpt55.json"},
    "Opus_4.7": {"eval": f"{EMAIL_RESULT}/commercial_llm/eval_opus47.json"},
    "Base_4B": {"eval": f"{EMAIL_RESULT}/ours_sft/eval_base.json"},
    "SFT_4B": {"eval": f"{EMAIL_RESULT}/ours_sft/eval_sft.json"},
    "GRPO_4B_step843": {"eval": f"{EMAIL_RESULT}/ours_rl/eval_grpo_step843.json"},
    "DAPO_4B_step843": {"eval": f"{EMAIL_RESULT}/ours_rl/eval_dapo_step843.json"},
}

def process_all(name, methods, gt_path):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    queries = load_gt(gt_path)
    e1, e3, e5 = {}, {}, {}
    for mname, info in methods.items():
        if "pred" in info:
            if not os.path.exists(info["pred"]):
                print(f"  SKIP {mname}: pred not found")
                continue
            pred = load_pred(info["pred"])
            overall = compute_metrics(pred, queries)
            by_level = compute_by_group(pred, queries, "level")
            by_mod = compute_by_group(pred, queries, "modality")
        elif "eval" in info:
            if not os.path.exists(info["eval"]):
                print(f"  SKIP {mname}: eval not found")
                continue
            overall = get_dsl_field(info["eval"], "overall")
            by_level = get_dsl_field(info["eval"], "by_level")
            by_mod = get_dsl_field(info["eval"], "by_modality")
        else:
            continue
        if overall:
            e1[mname] = overall
            print(f"  {mname}: Hit@1={overall['hit1']:.4f}  N={overall['n']}")
        if by_level:
            e3[mname] = by_level
        if by_mod:
            e5[mname] = by_mod
    return e1, e3, e5

ecom_e1, ecom_e3, ecom_e5 = process_all("Ecommerce", ECOM_METHODS, ECOM_GT)
email_e1, email_e3, email_e5 = process_all("Email", EMAIL_METHODS, EMAIL_GT)

# Save E1
e1 = {"ecommerce": ecom_e1, "email": email_e1}
json.dump(e1, open(f"{OUT_DIR}/e1_main_results.json", "w"), indent=2, ensure_ascii=False)
print(f"\nE1 saved ({len(ecom_e1)}+{len(email_e1)} methods)")

# Save E3
e3 = {"ecommerce": ecom_e3, "email": email_e3}
json.dump(e3, open(f"{OUT_DIR}/e3_complexity_breakdown.json", "w"), indent=2, ensure_ascii=False)
print(f"E3 saved ({len(ecom_e3)}+{len(email_e3)} methods)")

# Save E5
e5 = {"ecommerce": ecom_e5, "email": email_e5}
json.dump(e5, open(f"{OUT_DIR}/e5_modality_ablation.json", "w"), indent=2, ensure_ascii=False)
print(f"E5 saved ({len(ecom_e5)}+{len(email_e5)} methods)")

# ============================================================
# E6: DSL Orchestration Analysis
# ============================================================

def find_dsl_predictions(result_dir, results_dir):
    """Find DSL prediction files (contain string DSL, not ranked lists)."""
    files = {}
    for subdir in ["ours_rl", "ours_sft", "commercial_llm", "b5_fewshot_dsl"]:
        path = f"{result_dir}/{subdir}"
        if not os.path.isdir(path):
            continue
        for f in os.listdir(path):
            if f.startswith("predictions_") and f.endswith(".json"):
                files[f.replace("predictions_", "").replace(".json", "")] = os.path.join(path, f)
    if os.path.isdir(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith("predictions_") and f.endswith(".json"):
                label = f.replace("predictions_", "").replace(".json", "")
                if label not in files:
                    files[label] = os.path.join(results_dir, f)
    return files

def parse_pred_file(pred_path):
    """Load prediction file and normalize to {qid: dsl_dict_or_string}."""
    if not os.path.exists(pred_path):
        return None
    raw = json.load(open(pred_path))
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, dict) and "query_id" in item:
                result[item["query_id"]] = item.get("dsl", item)
        return result if result else None
    elif isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, (str, dict)):
            return raw
    return None

def extract_dsl_text(dsl_val):
    """Extract a searchable text representation from a DSL value."""
    if isinstance(dsl_val, str):
        return dsl_val
    if isinstance(dsl_val, dict):
        parts = []
        if "sql" in dsl_val:
            parts.append(dsl_val["sql"])
        for r in dsl_val.get("retrieval_list", []):
            parts.append(f"type={r.get('type','')} field={r.get('field','')} query={r.get('query','')}")
        return " ".join(parts)
    return ""

def analyze_dsl(pred_path):
    preds = parse_pred_file(pred_path)
    if not preds:
        return None
    action_counts = Counter()
    retrieval_type_counts = Counter()
    n_total = n_dsl = 0
    actions_per_q = []
    for qid, val in preds.items():
        n_total += 1
        dsl_text = extract_dsl_text(val)
        if not dsl_text:
            continue
        n_dsl += 1
        acts = set()
        if re.search(r'WHERE|SELECT|FROM', dsl_text, re.IGNORECASE):
            acts.add("sql_filter")
        if isinstance(val, dict):
            for r in val.get("retrieval_list", []):
                rtype = r.get("type", "")
                if rtype == "text":
                    acts.add("text_search")
                    retrieval_type_counts["text"] += 1
                elif rtype == "image":
                    acts.add("image_search")
                    retrieval_type_counts["image"] += 1
        else:
            if re.search(r'text_search|TEXT_SEARCH|semantic_search', dsl_text, re.IGNORECASE):
                acts.add("text_search")
            if re.search(r'image_search|IMAGE_SEARCH|visual_search', dsl_text, re.IGNORECASE):
                acts.add("image_search")
        if re.search(r'INTERSECT', dsl_text, re.IGNORECASE):
            acts.add("intersect")
        if re.search(r'UNION', dsl_text, re.IGNORECASE):
            acts.add("union")
        for a in acts:
            action_counts[a] += 1
        actions_per_q.append(len(acts))
    if n_dsl == 0:
        return None
    return {
        "n_total": n_total, "n_dsl": n_dsl,
        "action_freq": {k: round(v/n_dsl, 4) for k, v in action_counts.most_common()},
        "retrieval_types": {k: round(v/n_dsl, 4) for k, v in retrieval_type_counts.most_common()},
        "avg_actions": round(sum(actions_per_q)/len(actions_per_q), 2),
    }

e6 = {}
for domain, result_dir, results_dir in [
    ("ecommerce", ECOM_RESULT, f"{BASE}/results/ecommerce"),
    ("email", EMAIL_RESULT, f"{BASE}/results/email"),
]:
    e6[domain] = {}
    dsl_files = find_dsl_predictions(result_dir, results_dir)
    for label, path in sorted(dsl_files.items()):
        r = analyze_dsl(path)
        if r:
            e6[domain][label] = r
            print(f"  E6 {domain}/{label}: {r['n_dsl']} DSLs, avg_actions={r['avg_actions']}")

json.dump(e6, open(f"{OUT_DIR}/e6_orchestration_analysis.json", "w"), indent=2, ensure_ascii=False)
print(f"E6 saved")

# ============================================================
# E7: Modality Selection Analysis
# ============================================================

def analyze_mod_selection(pred_path, queries):
    preds = parse_pred_file(pred_path)
    if not preds:
        return None
    qmap = {q["query_id"]: q for q in queries}
    correct = total = 0
    confusion = defaultdict(lambda: Counter())
    for qid, val in preds.items():
        dsl_text = extract_dsl_text(val)
        if not dsl_text or qid not in qmap:
            continue
        q = qmap[qid]
        gt_mods = set(q.get("modalities", []))
        total += 1
        pred_mods = set()
        if isinstance(val, dict):
            for r in val.get("retrieval_list", []):
                rtype = r.get("type", "")
                if rtype == "text":
                    pred_mods.add("text")
                elif rtype == "image":
                    pred_mods.add("image")
            if re.search(r'WHERE|SELECT', val.get("sql", ""), re.IGNORECASE):
                pred_mods.add("structured")
        else:
            if re.search(r'text_search|TEXT_SEARCH|semantic_search', dsl_text, re.IGNORECASE):
                pred_mods.add("text")
            if re.search(r'image_search|IMAGE_SEARCH|visual_search', dsl_text, re.IGNORECASE):
                pred_mods.add("image")
            if re.search(r'SELECT|WHERE|brand|color|price|category|rating|sender|recipient|date|subject', dsl_text, re.IGNORECASE):
                pred_mods.add("structured")
        gt_key = "+".join(sorted(gt_mods))
        pred_key = "+".join(sorted(pred_mods))
        confusion[gt_key][pred_key] += 1
        if gt_mods == pred_mods:
            correct += 1
    if total == 0:
        return None
    return {
        "total": total,
        "accuracy": round(correct/total, 4),
        "confusion": {k: dict(v.most_common(5)) for k, v in confusion.items()},
    }

e7 = {}
ecom_queries = load_gt(ECOM_GT)
email_queries = load_gt(EMAIL_GT)
for domain, result_dir, results_dir, queries in [
    ("ecommerce", ECOM_RESULT, f"{BASE}/results/ecommerce", ecom_queries),
    ("email", EMAIL_RESULT, f"{BASE}/results/email", email_queries),
]:
    e7[domain] = {}
    dsl_files = find_dsl_predictions(result_dir, results_dir)
    for label, path in sorted(dsl_files.items()):
        r = analyze_mod_selection(path, queries)
        if r:
            e7[domain][label] = r
            print(f"  E7 {domain}/{label}: accuracy={r['accuracy']}")

json.dump(e7, open(f"{OUT_DIR}/e7_modality_selection.json", "w"), indent=2, ensure_ascii=False)
print(f"E7 saved")

# ============================================================
# E8: Model Scaling
# ============================================================

e8 = {"ecommerce": {}, "email": {}}
for label, path in [
    ("SFT_1.7B", f"{ECOM_RESULT}/ours_sft/eval_sft_1.7b.json"),
    ("SFT_4B", f"{ECOM_RESULT}/ours_sft/eval_sft4b.json"),
    ("SFT_8B", f"{ECOM_RESULT}/ours_sft/eval_sft8b.json"),
    ("Base_4B", f"{ECOM_RESULT}/ours_sft/eval_base4b.json"),
    ("Base_8B", f"{ECOM_RESULT}/ours_sft/eval_base8b.json"),
    ("GRPO_4B_step300", f"{ECOM_RESULT}/ours_rl/eval_grpo4b_step300.json"),
    ("GRPO_8B_step150", f"{ECOM_RESULT}/ours_rl/eval_grpo8b_step150.json"),
    ("DAPO_4B_step420", f"{ECOM_RESULT}/ours_rl/eval_dapo4b_step420.json"),
]:
    o = get_dsl_field(path, "overall")
    bl = get_dsl_field(path, "by_level")
    if o:
        e8["ecommerce"][label] = {"overall": o, "by_level": bl}

for label, path in [
    ("Base_4B", f"{EMAIL_RESULT}/ours_sft/eval_base.json"),
    ("SFT_4B", f"{EMAIL_RESULT}/ours_sft/eval_sft.json"),
    ("GRPO_4B_step843", f"{EMAIL_RESULT}/ours_rl/eval_grpo_step843.json"),
    ("DAPO_4B_step843", f"{EMAIL_RESULT}/ours_rl/eval_dapo_step843.json"),
]:
    o = get_dsl_field(path, "overall")
    bl = get_dsl_field(path, "by_level")
    if o:
        e8["email"][label] = {"overall": o, "by_level": bl}

json.dump(e8, open(f"{OUT_DIR}/e8_model_scaling.json", "w"), indent=2, ensure_ascii=False)
print(f"E8 saved ({len(e8['ecommerce'])}+{len(e8['email'])} methods)")

# ============================================================
# E9: Error Analysis
# ============================================================

def analyze_errors(eval_path):
    if not os.path.exists(eval_path):
        return None
    d = json.load(open(eval_path))
    if "overall" not in d:
        return None
    o = d["overall"]
    total = o.get("total") or o.get("count", 0)
    if total == 0:
        return None
    errs = d.get("errors", {})
    return {
        "total": total,
        "format_success": o.get("format_success", total),
        "exec_success": o.get("exec_success", total),
        "exec_rate": round(o.get("exec_success_rate", 1.0), 4),
        "error_types": {k: len(v) if isinstance(v, list) else v for k, v in errs.items()} if errs else {},
    }

e9 = {"ecommerce": {}, "email": {}}
for label, path in {
    "GRPO_4B_step300": f"{ECOM_RESULT}/ours_rl/eval_grpo4b_step300.json",
    "DAPO_4B_step420": f"{ECOM_RESULT}/ours_rl/eval_dapo4b_step420.json",
    "SFT_4B": f"{ECOM_RESULT}/ours_sft/eval_sft4b.json",
    "SFT_1.7B": f"{ECOM_RESULT}/ours_sft/eval_sft_1.7b.json",
    "Base_4B": f"{ECOM_RESULT}/ours_sft/eval_base4b.json",
    "GPT-5.4": f"{ECOM_RESULT}/commercial_llm/eval_gpt54.json",
    "GPT-5.5": f"{ECOM_RESULT}/commercial_llm/eval_gpt55.json",
    "Opus_4.7": f"{ECOM_RESULT}/commercial_llm/eval_opus47.json",
    "B5_fewshot_GPT55": f"{ECOM_RESULT}/b5_fewshot_dsl/eval_predictions_fewshot_gpt55.json",
    "B5_fewshot_Opus47": f"{ECOM_RESULT}/b5_fewshot_dsl/eval_predictions_fewshot_opus47.json",
}.items():
    r = analyze_errors(path)
    if r:
        e9["ecommerce"][label] = r

for label, path in {
    "DAPO_4B_step843": f"{EMAIL_RESULT}/ours_rl/eval_dapo_step843.json",
    "GRPO_4B_step843": f"{EMAIL_RESULT}/ours_rl/eval_grpo_step843.json",
    "SFT_4B": f"{EMAIL_RESULT}/ours_sft/eval_sft.json",
    "Base_4B": f"{EMAIL_RESULT}/ours_sft/eval_base.json",
    "GPT-5.4": f"{EMAIL_RESULT}/commercial_llm/eval_gpt54.json",
    "GPT-5.5": f"{EMAIL_RESULT}/commercial_llm/eval_gpt55.json",
    "Opus_4.7": f"{EMAIL_RESULT}/commercial_llm/eval_opus47.json",
}.items():
    r = analyze_errors(path)
    if r:
        e9["email"][label] = r

json.dump(e9, open(f"{OUT_DIR}/e9_error_analysis.json", "w"), indent=2, ensure_ascii=False)
print(f"E9 saved ({len(e9['ecommerce'])}+{len(e9['email'])} methods)")

# ============================================================
print(f"\n{'='*60}")
print(f"ALL DONE — results in {OUT_DIR}/")
print(f"  e1_main_results.json")
print(f"  e3_complexity_breakdown.json")
print(f"  e5_modality_ablation.json")
print(f"  e6_orchestration_analysis.json")
print(f"  e7_modality_selection.json")
print(f"  e8_model_scaling.json")
print(f"  e9_error_analysis.json")
print(f"{'='*60}")
