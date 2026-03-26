#!/usr/bin/env python3
"""
Step 9: Generate natural_language_query for each DSL query using DeepSeek (few-shot).

Input:  /data/youchengsong/DSL-R1/data/ecommerce/train/queries_product_train_step8.jsonl
Output: /data/youchengsong/DSL-R1/data/ecommerce/train/queries_product_train_step9.jsonl

Resume: intermediate results saved to _tmp_nl_product_train.jsonl

Usage:
    python gen_nl_query_product.py --api_key YOUR_KEY
    python gen_nl_query_product.py --api_key YOUR_KEY --test 20
    python gen_nl_query_product.py --api_key YOUR_KEY --concurrency 40
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/data/youchengsong/DSL-R1/data/ecommerce/train")
INPUT_FILE  = DATA_DIR / "queries_product_train_step8.jsonl"
TMP_FILE    = DATA_DIR / "_tmp_nl_product_train.jsonl"
OUTPUT_FILE = DATA_DIR / "queries_product_train_step9.jsonl"

API_BASE    = "https://token.claude-code-manager.com/v1"
MODEL       = "deepseek-v3-2-251201"
CONCURRENCY = 30

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at converting structured product search queries into natural, "
    "fluent English search queries that a real customer might type on an e-commerce platform."
)

# Few-shot examples covering: single field, AND, OR, NOT, cross-modal, complex
FEW_SHOT_EXAMPLES = [
    {
        "dsl": (
            "SQL: WHERE category = 'Running Shoes'\n"
            "Conditions:\n"
            "  category = 'Running Shoes'\n"
            "Structure: single condition"
        ),
        "nl": "running shoes",
    },
    {
        "dsl": (
            "SQL: WHERE brand = 'The North Face' AND average_rating = 4.5\n"
            "Conditions:\n"
            "  brand = 'The North Face'\n"
            "  average_rating = 4.5\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "The North Face products rated 4.5 stars",
    },
    {
        "dsl": (
            "SQL: WHERE product_id IN <text_0> AND price >= 23.75 AND price <= 26.25\n"
            "Conditions:\n"
            "  text matches: \"lightweight waterproof hiking backpack\"\n"
            "  price between 23.75 and 26.25\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "lightweight waterproof hiking backpack priced around $25",
    },
    {
        "dsl": (
            "SQL: WHERE product_id IN <image_0> OR brand = \"Levi's\"\n"
            "Conditions:\n"
            "  image looks like: \"classic blue denim jacket with button front\"\n"
            "  brand = \"Levi's\"\n"
            "Structure: OR (any condition can match)"
        ),
        "nl": "something that looks like a classic blue denim jacket, or any Levi's item",
    },
    {
        "dsl": (
            "SQL: WHERE brand != 'Nike' AND category = 'Athletic Footwear'\n"
            "Conditions:\n"
            "  brand != 'Nike'  [NOT]\n"
            "  category = 'Athletic Footwear'\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "athletic footwear that is not from Nike",
    },
    {
        "dsl": (
            "SQL: WHERE (product_id IN <text_0> OR (rating_number >= 374 AND rating_number <= 413) "
            "OR average_rating = 4.4) AND (brand != 'Recon Medical' OR category = 'Cell Phones & Accessories')\n"
            "Conditions:\n"
            "  Group 1 (OR):\n"
            "    text matches: \"Galaxy S10 case PU leather slim protective\"\n"
            "    rating_number between 374 and 413\n"
            "    average_rating = 4.4\n"
            "  Group 2 (OR):\n"
            "    brand != 'Recon Medical'  [NOT]\n"
            "    category = 'Cell Phones & Accessories'\n"
            "Structure: AND between groups, OR within groups"
        ),
        "nl": (
            "Galaxy S10 leather slim case with around 400 reviews and 4.4 stars, "
            "preferably not from Recon Medical or in the Cell Phones & Accessories category"
        ),
    },
    {
        "dsl": (
            "SQL: WHERE product_id IN <text_0> AND product_id IN <image_1> AND "
            "price >= 9.5 AND price <= 10.5\n"
            "Conditions:\n"
            "  text matches: \"scented soy candle vanilla relaxation\"\n"
            "  image looks like: \"small glass jar candle with wooden lid\"\n"
            "  price between 9.5 and 10.5\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "scented soy vanilla candle in a small glass jar with a wooden lid, priced between $9.5 and $10.5",
    },
]

FEW_SHOTS_STR = "\n\n".join(
    f"DSL:\n{ex['dsl']}\nNatural language query: {ex['nl']}"
    for ex in FEW_SHOT_EXAMPLES
)

USER_PROMPT_TEMPLATE = """\
Convert the following product search DSL into a natural, fluent English search query.

Rules:
- Preserve all exact numerical values and ranges as-is (e.g. "price between $0.49 and $0.55", not "around 50 cents").
- Always add a $ sign before price values (e.g. "priced between $40.41 and $44.67").
- For NOT conditions, use phrasing like "not X" or "excluding X".

{few_shots}

Now convert this DSL:
{dsl_repr}

Output ONLY the natural language query (1-2 sentences). No explanation, no quotes."""


# ── DSL formatter ─────────────────────────────────────────────────────────────
def format_leaf(leaf, retrieval_map):
    """retrieval_map: leaf_id -> retrieval query string (for text/image leaves)"""
    field    = leaf["field"]
    is_pos   = leaf["is_pos"]
    modality = leaf["modality"]
    value    = leaf["value"]

    if modality == "text":
        # Use the actual retrieval query (text_pos of gold), not the stored neg description
        v = retrieval_map.get(leaf["leaf_id"], value.get("val", ""))
        not_tag = " [NOT]" if not is_pos else ""
        return f'  text{not_tag}: "{v}"'
    elif modality == "image":
        v = retrieval_map.get(leaf["leaf_id"], value.get("val", ""))
        return f'  image looks like: "{v}"'
    else:
        # Structured leaves: always positive SQL conditions regardless of is_pos.
        # is_pos=false only means the value was sampled from a non-gold product.
        op  = value.get("op", "=")
        val = value.get("val")
        lo  = value.get("lo")
        hi  = value.get("hi")
        if lo is not None and hi is not None:
            return f"  {field} between {lo} and {hi}"
        else:
            return f"  {field} {op} {repr(val)}"


def build_retrieval_map(query):
    """Map leaf_id -> retrieval query string.
    text_0/text_1/... map to text leaves in leaf_id order.
    image_0/image_1/... map to image leaves in leaf_id order.
    """
    retrieval_list = query["dsl"].get("retrieval_list", [])
    text_queries  = [r["query"] for r in retrieval_list if r["type"] == "text"]
    image_queries = [r["query"] for r in retrieval_list if r["type"] == "image"]

    text_leaf_ids  = sorted(lf["leaf_id"] for lf in query["leaves"] if lf["modality"] == "text")
    image_leaf_ids = sorted(lf["leaf_id"] for lf in query["leaves"] if lf["modality"] == "image")

    rmap = {}
    for i, lid in enumerate(text_leaf_ids):
        if i < len(text_queries):
            rmap[lid] = text_queries[i]
    for i, lid in enumerate(image_leaf_ids):
        if i < len(image_queries):
            rmap[lid] = image_queries[i]
    return rmap


def dsl_to_repr(query):
    sql          = query["dsl"]["sql"]
    leaves       = {lf["leaf_id"]: lf for lf in query["leaves"]}
    groups       = query["groups"]
    inner_op     = query.get("inner_op") or "AND"
    outer_op     = query.get("outer_op")
    k            = len(groups)
    retrieval_map = build_retrieval_map(query)

    lines = [f"SQL: {sql}", "Conditions:"]

    if k == 1 or outer_op is None:
        for lid in groups[0]:
            lines.append(format_leaf(leaves[lid], retrieval_map))
        if len(groups[0]) == 1:
            struct = "single condition"
        elif inner_op == "AND":
            struct = "AND (all conditions must match)"
        else:
            struct = "OR (any condition can match)"
    else:
        for gi, group in enumerate(groups):
            if len(group) == 1:
                lines.append(format_leaf(leaves[group[0]], retrieval_map))
            else:
                lines.append(f"  Group {gi + 1} ({inner_op}):")
                for lid in group:
                    lines.append("  " + format_leaf(leaves[lid], retrieval_map))
        struct = f"AND between groups, {inner_op} within groups" \
                 if outer_op == "AND" else f"OR between groups, {inner_op} within groups"

    lines.append(f"Structure: {struct}")
    return "\n".join(lines)


# ── API call ──────────────────────────────────────────────────────────────────
async def gen_nl_one(client, sem, query_id, dsl_repr):
    async with sem:
        for attempt in range(3):
            try:
                user_msg = USER_PROMPT_TEMPLATE.format(
                    few_shots=FEW_SHOTS_STR,
                    dsl_repr=dsl_repr,
                )
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=150,
                    temperature=0.4,
                )
                msg = resp.choices[0].message if resp.choices else None
                if not msg or not msg.content:
                    raise ValueError("empty response")
                nl = msg.content.strip().strip('"').strip("'")
                return query_id, nl
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED {query_id}: {e}")
                    return query_id, None
                await asyncio.sleep(2 ** attempt)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",     default=os.environ.get("API_KEY", ""))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--test",        type=int, default=0,
                        help="Only process N queries (for testing)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Set --api_key or env var API_KEY")

    # Load all queries
    print(f"Loading {INPUT_FILE} ...")
    queries = []
    with open(INPUT_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Total queries: {len(queries)}")

    if args.test > 0:
        queries = queries[:args.test]
        print(f"Test mode: {len(queries)} queries")

    # Resume: load already processed from tmp file
    done = {}
    if TMP_FILE.exists():
        with open(TMP_FILE) as f:
            for line in f:
                obj = json.loads(line)
                done[obj["query_id"]] = obj["natural_language_query"]
    print(f"Already done: {len(done)}")

    # Generate for all queries
    remaining = [q for q in queries if q["query_id"] not in done]
    print(f"Remaining: {len(remaining)}")

    if remaining:
        client = AsyncOpenAI(base_url=API_BASE, api_key=args.api_key)
        sem    = asyncio.Semaphore(args.concurrency)
        coros  = [gen_nl_one(client, sem, q["query_id"], dsl_to_repr(q)) for q in remaining]

        fout = open(TMP_FILE, "a", encoding="utf-8")
        with tqdm(total=len(coros), desc="gen_nl_query") as pbar:
            for coro in asyncio.as_completed(coros):
                qid, nl = await coro
                if nl:
                    done[qid] = nl
                    fout.write(json.dumps({"query_id": qid, "natural_language_query": nl},
                                          ensure_ascii=False) + "\n")
                    fout.flush()
                pbar.update(1)
        fout.close()

    # Merge NL queries back into full query list and write step9 output
    print(f"\nMerging into {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in queries:
            q["natural_language_query"] = done.get(q["query_id"])
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    ok = sum(1 for q in queries if q.get("natural_language_query"))
    print(f"Done. {ok}/{len(queries)} queries have natural_language_query → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
