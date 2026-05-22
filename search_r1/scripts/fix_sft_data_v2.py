"""
Fix SFT data for Search-R1 and SQL-R1 baselines.
v2: Fixed numpy array handling, correct prompt parsing, proper email schema.
"""
import json, os, random, re, sys, math
import pandas as pd
from collections import defaultdict

random.seed(42)

BASE = "/root/autodl-tmp"

class SimpleBM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_len = []
        self.tf = []
        self.df = defaultdict(int)
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
    lines = []
    for doc, score in results:
        lines.append(f"[{doc['id']}] {doc.get('title','')} - {doc.get('text','')[:200]}")
    return "\n".join(lines) if lines else "No results found."


def parse_prompt(raw_prompt):
    """Convert RL parquet prompt (numpy array or str or list) to list of dicts."""
    if hasattr(raw_prompt, 'tolist'):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        raw_prompt = json.loads(raw_prompt)
    return raw_prompt


def extract_user_query_search(prompt_msgs):
    """Extract user query from Search-R1 RL prompt (single user msg with 'Query: ...' at end)."""
    content = prompt_msgs[-1]["content"]
    m = re.search(r'Query:\s*(.+)', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content[-200:].strip()


def extract_user_query_sql(prompt_msgs):
    """Extract user query from SQL-R1 RL prompt (system + user msgs, user msg IS the query)."""
    for msg in reversed(prompt_msgs):
        if msg["role"] == "user":
            return msg["content"].strip()
    return prompt_msgs[-1]["content"][-200:].strip()


def extract_search_query_from_dsl(assistant_content):
    """Extract search query from source SFT DSL answer."""
    try:
        m = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', assistant_content, re.DOTALL)
        if m:
            dsl = json.loads(m.group(1))
            ret_list = dsl.get("retrieval_list", [])
            if ret_list and ret_list[0].get("query"):
                return ret_list[0]["query"]
    except:
        pass
    return None


def extract_sql_from_dsl(assistant_content, domain):
    """Extract SQL conditions from source SFT DSL answer."""
    try:
        m = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', assistant_content, re.DOTALL)
        if m:
            dsl = json.loads(m.group(1))
            sql_cond = dsl.get("sql", "")
            if sql_cond:
                table = "products" if domain == "ecommerce" else "emails"
                id_col = "product_id" if domain == "ecommerce" else "message_id"
                # Clean up placeholders like <image_0>, <text_0>
                sql_cond = re.sub(r'<\w+_\d+>', f"(SELECT {id_col} FROM {table} LIMIT 10)", sql_cond)
                if "WHERE" in sql_cond.upper():
                    return f"SELECT DISTINCT {id_col} FROM {table} {sql_cond}"
                else:
                    return f"SELECT DISTINCT {id_col} FROM {table} WHERE {sql_cond}"
    except:
        pass
    return None


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
    "Based on the search results, {pid} appears to be the most relevant match for this query.",
    "Looking at the search results, I can see that {pid} best matches what the user is looking for.",
    "From the search results, {pid} is the most relevant product for this query.",
    "After reviewing the results, {pid} seems to be the best match.",
    "The search results show that {pid} is the most relevant item.",
    "Based on these results, I'll go with {pid} as the best match.",
    "Analyzing the search results, {pid} matches the user's requirements best.",
    "The most relevant product from the search results is {pid}.",
]

THINK_SQL_TEMPLATES = [
    "I need to understand the user's query and formulate a SQL query to find matching items.",
    "Let me analyze what the user is looking for and write a SQL query.",
    "I should construct a SQL query based on the user's requirements.",
    "Let me think about the right SQL conditions for this query.",
    "I need to translate the user's natural language query into SQL.",
    "Let me figure out the right SQL filters for this search.",
]

THINK_SQL_ANSWER = [
    "This SQL query filters based on the user's criteria and should return the relevant results.",
    "This query captures the key conditions from the user's request.",
    "The SQL query targets the specific attributes mentioned by the user.",
    "This should correctly filter based on the given requirements.",
]


def fix_search_r1_sft(domain):
    print(f"\n=== Fixing Search-R1 SFT for {domain} ===")

    rl_df = pd.read_parquet(f"{BASE}/search_r1/rl_data/{domain}_search_rl_train.parquet")
    print(f"  RL data: {len(rl_df)} rows")

    sft_rows = []
    with open(f"{BASE}/data/{domain}/sft_train.jsonl") as f:
        for line in f:
            sft_rows.append(json.loads(line))
    print(f"  Source SFT: {len(sft_rows)} rows")

    print(f"  Loading corpus and building BM25 index...")
    corpus = load_corpus(domain)
    bm25 = SimpleBM25(corpus)
    print(f"  Corpus: {len(corpus)} docs, BM25 ready")

    corpus_by_id = {doc["id"]: doc for doc in corpus}

    system_prompt = (
        "You are a helpful search assistant. Given a user query, find the most relevant product by searching the product catalog.\n"
        "You must conduct reasoning inside <think> and </think> first every time you get new information.\n"
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.\n"
        "You can search as many times as you want.\n"
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations."
    )

    out_path = f"{BASE}/search_r1/sft_data/{domain}_search_sft.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    count = 0
    skipped = 0
    bm25_empty = 0
    with open(out_path, "w") as fout:
        for idx in range(min(len(rl_df), len(sft_rows))):
            rm = rl_df.iloc[idx]["reward_model"]
            if isinstance(rm, str):
                rm = json.loads(rm)
            gt = rm.get("ground_truth", {})
            target_ids = gt.get("target", [])
            if not target_ids:
                skipped += 1
                continue

            prompt_msgs = parse_prompt(rl_df.iloc[idx]["prompt"])
            user_query = extract_user_query_search(prompt_msgs)

            # Get search query from source DSL
            src_assistant = sft_rows[idx]["messages"][-1]["content"]
            search_query = extract_search_query_from_dsl(src_assistant)
            if not search_query:
                search_query = user_query[:100]

            # Run BM25 search
            results = bm25.search(search_query, top_k=5)
            if not results:
                results = bm25.search(user_query[:80], top_k=5)
                bm25_empty += 1

            info_block = format_search_results(results)

            answer_id = target_ids[0]
            result_ids = [doc["id"] for doc, _ in results]
            for tid in target_ids:
                if tid in result_ids:
                    answer_id = tid
                    break

            think1 = random.choice(THINK_SEARCH_TEMPLATES)
            think2 = random.choice(THINK_ANSWER_TEMPLATES).format(pid=answer_id)

            assistant_response = (
                f"<think>\n{think1}\n</think>\n\n"
                f"<search>{search_query}</search>\n\n"
                f"<information>\n{info_block}\n</information>\n\n"
                f"<think>\n{think2}\n</think>\n\n"
                f"<answer>{answer_id}</answer>"
            )

            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
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

    print(f"\n  Written {count} samples to {out_path} (skipped {skipped} no GT, {bm25_empty} BM25 fallback)")
    return out_path


def fix_sql_r1_sft(domain):
    print(f"\n=== Fixing SQL-R1 SFT for {domain} ===")

    sft_rows = []
    with open(f"{BASE}/data/{domain}/sft_train.jsonl") as f:
        for line in f:
            sft_rows.append(json.loads(line))
    print(f"  Source SFT: {len(sft_rows)} rows")

    rl_df = pd.read_parquet(f"{BASE}/sql_r1/rl_data/{domain}_sql_rl_train.parquet")
    print(f"  RL data: {len(rl_df)} rows")

    # Get the actual system prompt from RL data (includes correct schema)
    first_prompt = parse_prompt(rl_df.iloc[0]["prompt"])
    sql_system = first_prompt[0]["content"]  # system msg with schema
    print(f"  System prompt length: {len(sql_system)} chars")

    table = "products" if domain == "ecommerce" else "emails"
    id_col = "product_id" if domain == "ecommerce" else "message_id"
    fallback_sql = f"SELECT DISTINCT {id_col} FROM {table}"

    out_path = f"{BASE}/sql_r1/sft_data/{domain}_sql_sft.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    count = 0
    fallback_count = 0
    with open(out_path, "w") as fout:
        for idx in range(min(len(rl_df), len(sft_rows))):
            prompt_msgs = parse_prompt(rl_df.iloc[idx]["prompt"])
            user_query = extract_user_query_sql(prompt_msgs)

            # Extract SQL from source DSL
            src_assistant = sft_rows[idx]["messages"][-1]["content"]
            sql_query = extract_sql_from_dsl(src_assistant, domain)
            if not sql_query:
                sql_query = fallback_sql
                fallback_count += 1

            think1 = random.choice(THINK_SQL_TEMPLATES)
            think2 = random.choice(THINK_SQL_ANSWER)

            assistant_response = (
                f"<think>\n{think1}\n</think>\n\n"
                f"<answer>\n```sql\n{sql_query}\n```\n</answer>"
            )

            row = {
                "messages": [
                    {"role": "system", "content": sql_system},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count <= 2:
                print(f"\n  Sample {count}:")
                print(f"    Query: {user_query[:100]}")
                print(f"    SQL: {sql_query[:150]}")

    print(f"\n  Written {count} samples to {out_path} ({fallback_count} used fallback SQL)")
    return out_path


if __name__ == "__main__":
    print("=== Generating fixed SFT data (v2) ===\n")

    for domain in ["ecommerce", "email"]:
        fix_search_r1_sft(domain)

    for domain in ["ecommerce", "email"]:
        fix_sql_r1_sft(domain)

    print("\n\n=== ALL DONE ===")
    print("Next: re-run SFT training with the new data, then RL.")
