#!/usr/bin/env python3
"""
Step 9: Generate natural_language_query for each email DSL query using DeepSeek (few-shot).

Input:  /data/youchengsong/DSL-R1/data/email/test/queries_email_test_step8.jsonl
Output: /data/youchengsong/DSL-R1/data/email/test/queries_email_test_step9.jsonl

Resume: intermediate results saved to _tmp_nl_email_test.jsonl

Usage:
    python gen_nl_query_email_test.py --api_key YOUR_KEY
    python gen_nl_query_email_test.py --api_key YOUR_KEY --test 10
    python gen_nl_query_email_test.py --api_key YOUR_KEY --concurrency 30
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/data/youchengsong/DSL-R1/data/email/test")
INPUT_FILE  = DATA_DIR / "queries_email_test_step8.jsonl"
TMP_FILE    = DATA_DIR / "_tmp_nl_email_test.jsonl"
OUTPUT_FILE = DATA_DIR / "queries_email_test_step9.jsonl"

API_BASE    = "https://token.claude-code-manager.com/v1"
MODEL       = "deepseek-v3-2-251201"
CONCURRENCY = 30

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at converting structured email search queries into natural, "
    "fluent English search queries that a real user might type in an email client."
)

# Few-shot examples covering: single field, AND, OR, NOT, Date, complex structures
FEW_SHOT_EXAMPLES = [
    {
        "dsl": (
            "SQL: WHERE message_id IN <text_0>\n"
            "Conditions:\n"
            "  Subject text: \"quarterly earnings report finance\"\n"
            "Structure: single condition"
        ),
        "nl": "emails with a subject about quarterly earnings report finance",
    },
    {
        "dsl": (
            "SQL: WHERE message_id IN <text_0>\n"
            "Conditions:\n"
            "  Message text: \"apology about missed summer vacation time off\"\n"
            "Structure: single condition"
        ),
        "nl": "emails containing an apology about missed summer vacation time off",
    },
    {
        "dsl": (
            "SQL: WHERE \"From\" = 'john.smith@enron.com' AND year = 2001\n"
            "Conditions:\n"
            "  From = 'john.smith@enron.com'\n"
            "  year = 2001\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "emails from john.smith@enron.com sent in 2001",
    },
    {
        "dsl": (
            "SQL: WHERE \"To\" = 'all.employees@enron.com' AND message_id IN <text_0>\n"
            "Conditions:\n"
            "  To = 'all.employees@enron.com'\n"
            "  Message text: \"merger announcement restructuring layoffs\"\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "emails sent to all.employees@enron.com about merger announcement restructuring layoffs",
    },
    {
        "dsl": (
            "SQL: WHERE message_id IN <text_0> OR \"From\" = 'marcus.nettelton@enron.com'\n"
            "Conditions:\n"
            "  Subject text: \"physical options exercise or trade\"\n"
            "  From = 'marcus.nettelton@enron.com'\n"
            "Structure: OR (any condition can match)"
        ),
        "nl": "emails with a subject about physical options exercise or trade, or any email from marcus.nettelton@enron.com",
    },
    {
        "dsl": (
            "SQL: WHERE (\"To\" = 'petroleumworldreaders@hotmail.com' OR "
            "\"From\" = 'petroleumworldnews11@hotmail.com') AND "
            "(message_id IN <text_0> OR message_id IN <text_1>)\n"
            "Conditions:\n"
            "  Group 1 (OR):\n"
            "    To = 'petroleumworldreaders@hotmail.com'\n"
            "    From = 'petroleumworldnews11@hotmail.com'\n"
            "  Group 2 (OR):\n"
            "    Subject text: \"petroleum industry weekly news update\"\n"
            "    Message text: \"Venezuela oil union strike protest new hydrocarbons law\"\n"
            "Structure: AND between groups, OR within groups"
        ),
        "nl": (
            "emails to petroleumworldreaders@hotmail.com or from petroleumworldnews11@hotmail.com, "
            "and whose subject is about petroleum industry weekly news or whose content mentions "
            "Venezuela oil union strike protest"
        ),
    },
    {
        "dsl": (
            "SQL: WHERE message_id NOT IN <text_0> AND \"From\" = 'jeff.dasovich@enron.com'\n"
            "Conditions:\n"
            "  Subject text [NOT]: \"internal meeting schedule agenda\"\n"
            "  From = 'jeff.dasovich@enron.com'\n"
            "Structure: AND (all conditions must match)"
        ),
        "nl": "emails from jeff.dasovich@enron.com that are not about internal meeting schedule agenda",
    },
    {
        "dsl": (
            "SQL: WHERE year >= 2000 AND year <= 2002\n"
            "Conditions:\n"
            "  year between 2000 and 2002\n"
            "Structure: single condition"
        ),
        "nl": "emails sent between 2000 and 2002",
    },
]

FEW_SHOTS_STR = "\n\n".join(
    f"DSL:\n{ex['dsl']}\nNatural language query: {ex['nl']}"
    for ex in FEW_SHOT_EXAMPLES
)

USER_PROMPT_TEMPLATE = """\
Convert the following email search DSL into a natural, fluent English search query.

Rules:
- Preserve all exact values as-is, including email addresses, years, and text queries.
- Preserve exact year values and ranges (e.g. "sent in 2001", "sent between 2000 and 2002"), do not round or approximate.
- For Subject text conditions, use phrasing like "with a subject about X".
- For Message text conditions, use phrasing like "containing X" or "about X".
- For NOT text conditions, use phrasing like "not about X" or "excluding messages about X".
- Structured leaves (From, To, Date/year) are always positive conditions regardless of context.

{few_shots}

Now convert this DSL:
{dsl_repr}

Output ONLY the natural language query (1-2 sentences). No explanation, no quotes."""


# ── DSL formatter ─────────────────────────────────────────────────────────────
def format_leaf(leaf, retrieval_map):
    """retrieval_map: leaf_id -> retrieval query string (for text leaves)"""
    field    = leaf["field"]
    is_pos   = leaf["is_pos"]
    modality = leaf["modality"]
    value    = leaf["value"]

    if modality == "text":
        v = retrieval_map.get(leaf["leaf_id"], value.get("val", ""))
        field_tag = f"{field} text"
        not_tag = " [NOT]" if not is_pos else ""
        return f"  {field_tag}{not_tag}: \"{v}\""
    else:
        op  = value.get("op", "=")
        val = value.get("val")
        lo  = value.get("lo")
        hi  = value.get("hi")
        if lo is not None and hi is not None:
            return f"  {field} between {lo} and {hi}"
        else:
            return f"  {field} {op} {repr(val)}"


def build_retrieval_map(query):
    retrieval_list = query["dsl"].get("retrieval_list", [])
    text_queries   = [r["query"] for r in retrieval_list if r["type"] == "text"]
    text_leaf_ids  = sorted(lf["leaf_id"] for lf in query["leaves"] if lf["modality"] == "text")

    rmap = {}
    for i, lid in enumerate(text_leaf_ids):
        if i < len(text_queries):
            rmap[lid] = text_queries[i]
    return rmap


def dsl_to_repr(query):
    sql           = query["dsl"]["sql"]
    leaves        = {lf["leaf_id"]: lf for lf in query["leaves"]}
    groups        = query["groups"]
    inner_op      = query.get("inner_op") or "AND"
    outer_op      = query.get("outer_op")
    k             = len(groups)
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
