"""
Fix SFT data for Search-R1 and SQL-R1 baselines.
- Uses real GT IDs from RL parquet as answers
- Runs BM25 against corpus for real search results
- Generates varied think blocks
"""
import json, os, random, re, sys
import pandas as pd
from collections import defaultdict

random.seed(42)

BASE = "/root/autodl-tmp"

# ============================================================
# BM25 search (simple TF-IDF based)
# ============================================================
class SimpleBM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs  # list of {"id":..., "title":..., "text":...}
        self.k1, self.b = k1, b
        self.doc_len = []
        self.tf = []  # per-doc term freq
        self.df = defaultdict(int)  # doc freq
        self.N = len(docs)
        for doc in docs:
            tokens = self._tokenize(doc.get("title","") + " " + doc.get("text",""))
            self.doc_len.append(len(tokens))
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.tf.append(tf)
            for t in set(tokens):
                self.df[t] += 1
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def search(self, query, top_k=5):
        q_tokens = self._tokenize(query)
        scores = []
        import math
        for i in range(self.N):
            score = 0.0
            dl = self.doc_len[i]
            for t in q_tokens:
                if t not in self.tf[i]:
                    continue
                tf_val = self.tf[i][t]
                df_val = self.df.get(t, 0)
                idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
                tf_norm = (tf_val * (self.k1 + 1)) / (tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tf_norm
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [(self.docs[i], s) for s, i in scores[:top_k]]


def load_corpus(domain):
    path = f"{BASE}/search_r1/corpus/{domain}_corpus.jsonl"
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def format_search_results(results):
    """Format BM25 results as the <information> block"""
    lines = []
    for doc, score in results:
        lines.append(f"[{doc['id']}] {doc.get('title','')} - {doc.get('text','')[:200]}")
    return "\n".join(lines)


# ============================================================
# Think block templates (varied)
# ============================================================
THINK_SEARCH_TEMPLATES = [
    "I need to search for products matching this query. Let me formulate a good search query.",
    "Let me search the product catalog to find what the user is looking for.",
    "I should search for relevant products based on the user's requirements.",
    "The user wants specific products. Let me search the catalog.",
    "I'll search for products that match the user's description.",
    "Let me find relevant products by searching the catalog.",
    "I need to look up products matching these criteria.",
    "Let me query the product database to find matching items.",
]

THINK_ANSWER_TEMPLATES = [
    "Based on the search results, {product_id} appears to be the most relevant match for this query.",
    "Looking at the search results, I can see that {product_id} best matches what the user is looking for.",
    "From the search results, {product_id} is the most relevant product for this query.",
    "After reviewing the results, {product_id} seems to be the best match.",
    "The search results show that {product_id} is the most relevant item.",
    "Based on these results, I'll go with {product_id} as the best match.",
    "Analyzing the search results, {product_id} matches the user's requirements best.",
    "The most relevant product from the search results is {product_id}.",
]


# ============================================================
# Fix Search-R1 SFT data
# ============================================================
def fix_search_r1_sft(domain):
    print(f"\n=== Fixing Search-R1 SFT for {domain} ===")

    # Load RL data to get queries and GT IDs
    rl_path = f"{BASE}/search_r1/rl_data/{domain}_search_rl_train.parquet"
    rl_df = pd.read_parquet(rl_path)
    print(f"  RL data: {len(rl_df)} rows")

    # Load source SFT data to get retrieval queries from DSL
    sft_src_path = f"{BASE}/data/{domain}/sft_train.jsonl"
    sft_rows = []
    with open(sft_src_path) as f:
        for line in f:
            sft_rows.append(json.loads(line))
    print(f"  Source SFT: {len(sft_rows)} rows")

    # Load corpus and build BM25
    print(f"  Loading corpus and building BM25 index...")
    corpus = load_corpus(domain)
    bm25 = SimpleBM25(corpus)
    print(f"  Corpus: {len(corpus)} docs, BM25 ready")

    # Build corpus ID lookup
    corpus_by_id = {doc["id"]: doc for doc in corpus}

    # Generate fixed SFT data
    out_path = f"{BASE}/search_r1/sft_data/{domain}_search_sft.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # System prompt for Search-R1
    system_prompt = """You are a helpful search assistant. Given a user query, find the most relevant product by searching the product catalog.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations."""

    count = 0
    skipped = 0
    with open(out_path, "w") as fout:
        for idx in range(min(len(rl_df), len(sft_rows))):
            # Get GT from RL data
            rm = rl_df.iloc[idx]["reward_model"]
            if isinstance(rm, str):
                rm = json.loads(rm)
            gt = rm.get("ground_truth", {})
            target_ids = gt.get("target", [])
            if not target_ids:
                skipped += 1
                continue

            # Get user query from RL prompt
            prompt = rl_df.iloc[idx]["prompt"]
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            if isinstance(prompt, list):
                user_msg = prompt[-1]["content"] if prompt else ""
            else:
                user_msg = str(prompt)

            # Extract just the query part (after "Query: ")
            query_match = re.search(r"Query:\s*(.+)", user_msg, re.DOTALL)
            user_query = query_match.group(1).strip() if query_match else user_msg[-200:]

            # Extract search query from source DSL
            src_assistant = sft_rows[idx]["messages"][-1]["content"]
            search_query = user_query[:100]  # default
            try:
                dsl_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', src_assistant, re.DOTALL)
                if dsl_match:
                    dsl = json.loads(dsl_match.group(1))
                    ret_list = dsl.get("retrieval_list", [])
                    if ret_list:
                        search_query = ret_list[0].get("query", search_query)
            except:
                pass

            # Run BM25 search
            results = bm25.search(search_query, top_k=5)
            info_block = format_search_results(results)

            # Pick answer: prefer GT ID that appears in results, else first GT
            answer_id = target_ids[0]
            result_ids = [doc["id"] for doc, _ in results]
            for tid in target_ids:
                if tid in result_ids:
                    answer_id = tid
                    break

            # Build assistant response with varied templates
            think1 = random.choice(THINK_SEARCH_TEMPLATES)
            think2 = random.choice(THINK_ANSWER_TEMPLATES).format(product_id=answer_id)

            assistant_response = f"""<think>
{think1}
</think>

<search>{search_query}</search>

<information>
{info_block}
</information>

<think>
{think2}
</think>

<answer>{answer_id}</answer>"""

            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {user_query}"},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count <= 2:
                print(f"\n  Sample {count}:")
                print(f"    Query: {user_query[:100]}")
                print(f"    Search: {search_query[:80]}")
                print(f"    Answer: {answer_id}")
                print(f"    Results: {result_ids[:3]}")

    print(f"\n  Written {count} samples to {out_path} (skipped {skipped} with no GT)")
    return out_path


# ============================================================
# Fix SQL-R1 SFT data
# ============================================================
THINK_SQL_SEARCH = [
    "I need to understand the user's query and formulate a SQL query to find matching products.",
    "Let me analyze what the user is looking for and write a SQL query.",
    "I should construct a SQL query based on the user's requirements.",
    "Let me think about the right SQL conditions for this query.",
    "I need to translate the user's natural language query into SQL.",
    "Let me figure out the right SQL filters for this search.",
]

THINK_SQL_ANSWER = [
    "This SQL query filters products based on the user's criteria and should return the relevant results.",
    "This query captures the key conditions from the user's request.",
    "The SQL query targets the specific attributes mentioned by the user.",
    "This should correctly filter the products based on the given requirements.",
]


def fix_sql_r1_sft(domain):
    print(f"\n=== Fixing SQL-R1 SFT for {domain} ===")

    # Load source SFT data
    sft_src_path = f"{BASE}/data/{domain}/sft_train.jsonl"
    sft_rows = []
    with open(sft_src_path) as f:
        for line in f:
            sft_rows.append(json.loads(line))
    print(f"  Source SFT: {len(sft_rows)} rows")

    # Load RL data for GT
    rl_path = f"{BASE}/sql_r1/rl_data/{domain}_sql_rl_train.parquet"
    rl_df = pd.read_parquet(rl_path)
    print(f"  RL data: {len(rl_df)} rows")

    # DB schema (extract from source SFT system prompt)
    src_system = sft_rows[0]["messages"][0]["content"]

    # SQL-R1 system prompt
    sql_system = """You are a SQL expert. Given a user query about products, write a SQL query to find matching items.
You must conduct reasoning inside <think> and </think> before writing your SQL.
Provide the final SQL query inside <answer> and </answer> tags.
The database has a 'products' table with columns: product_id, title, brand, category, price, rating, description, image_url."""

    out_path = f"{BASE}/sql_r1/sft_data/{domain}_sql_sft.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    count = 0
    with open(out_path, "w") as fout:
        for idx in range(min(len(rl_df), len(sft_rows))):
            # Get user query from RL prompt
            prompt = rl_df.iloc[idx]["prompt"]
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            if isinstance(prompt, list):
                user_msg = prompt[-1]["content"] if prompt else ""
            else:
                user_msg = str(prompt)
            query_match = re.search(r"Query:\s*(.+)", user_msg, re.DOTALL)
            user_query = query_match.group(1).strip() if query_match else user_msg[-200:]

            # Extract SQL from source DSL
            src_assistant = sft_rows[idx]["messages"][-1]["content"]
            sql_query = "SELECT DISTINCT product_id FROM products"  # fallback
            try:
                dsl_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', src_assistant, re.DOTALL)
                if dsl_match:
                    dsl = json.loads(dsl_match.group(1))
                    sql_cond = dsl.get("sql", "")
                    if sql_cond and "WHERE" in sql_cond.upper():
                        sql_query = f"SELECT DISTINCT product_id FROM products {sql_cond}"
                    elif sql_cond:
                        sql_query = f"SELECT DISTINCT product_id FROM products WHERE {sql_cond}"
            except:
                pass

            think1 = random.choice(THINK_SQL_SEARCH)
            think2 = random.choice(THINK_SQL_ANSWER)

            assistant_response = f"""<think>
{think1}
</think>

<answer>```sql
{sql_query}
```</answer>

<think>
{think2}
</think>"""

            row = {
                "messages": [
                    {"role": "system", "content": sql_system},
                    {"role": "user", "content": f"Query: {user_query}"},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count <= 2:
                print(f"\n  Sample {count}:")
                print(f"    Query: {user_query[:100]}")
                print(f"    SQL: {sql_query[:150]}")

    print(f"\n  Written {count} samples to {out_path}")
    return out_path


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Stop any running training first
    print("=== Generating fixed SFT data ===\n")

    # Search-R1
    for domain in ["ecommerce", "email"]:
        fix_search_r1_sft(domain)

    # SQL-R1
    for domain in ["ecommerce", "email"]:
        fix_sql_r1_sft(domain)

    print("\n\n=== ALL DONE ===")
    print("Next: re-run SFT training with the new data, then RL.")
