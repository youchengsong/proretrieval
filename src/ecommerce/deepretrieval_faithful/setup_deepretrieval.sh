#!/bin/bash
# One-shot setup for DeepRetrieval on 102 (A6000 server).
# Run this once before training.
set -e

PYTHON=~/miniconda3/envs/mlx/bin/python
WORK_DIR=~/LocalSearch/deepretrieval
DATA_SRC=~/LocalSearch/data/ecommerce
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

mkdir -p $WORK_DIR/data $WORK_DIR/index

echo "=== Step 1: Install pyserini ==="
$PYTHON -m pip install pyserini -q 2>/dev/null || true

echo "=== Step 2: Convert products to Pyserini JSONL ==="
$PYTHON -c "
import json, os

products_file = '$DATA_SRC/test/products_test.json'
out_dir = '$WORK_DIR/jsonl_docs'
os.makedirs(out_dir, exist_ok=True)

with open(products_file) as f:
    products = json.load(f)

docs = []
for p in products:
    pid = p.get('product_id', '')
    fields = []
    for k in ['brand', 'color', 'category', 'price', 'average_rating']:
        if k in p and p[k]:
            fields.append(f'{k}: {p[k]}')
    text_content = p.get('text', '')
    contents = ' '.join(fields) + ' ' + text_content
    docs.append({'id': pid, 'contents': contents.strip()})

with open(f'{out_dir}/docs.jsonl', 'w') as f:
    for d in docs:
        f.write(json.dumps(d) + '\n')

print(f'Converted {len(docs)} products to {out_dir}/docs.jsonl')
"

echo "=== Step 3: Build Lucene index ==="
$PYTHON -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $WORK_DIR/jsonl_docs \
    --index $WORK_DIR/index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4 \
    --storePositions --storeDocvectors --storeRaw

echo "=== Step 4: Verify index ==="
$PYTHON -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('$WORK_DIR/index')
print(f'Index: {s.num_docs} documents')
hits = s.search('Nike running shoes', k=5)
for h in hits:
    print(f'  {h.docid}: {h.score:.4f}')
"

echo "=== Step 5: Generate training data ==="
$PYTHON -c "
import json, os
from datasets import Dataset

SYSTEM_PROMPT = '''You are an expert in product search query generation. Given a natural language product search query, generate a Boolean search query using AND, OR, NOT operators and parentheses to retrieve relevant products via BM25.

The product database contains product titles, descriptions, brands, and categories as searchable text. Your Boolean query will be matched against all these fields.

Show your reasoning in <think> </think> tags. Then provide your Boolean query in JSON format within <answer> </answer> tags.

Example:
<think>
The user wants wireless noise-cancelling headphones. I should combine key terms.
</think>
<answer>
{\"query\": \"(\\\"wireless\\\" OR \\\"bluetooth\\\") AND \\\"headphones\\\" AND (\\\"noise cancelling\\\" OR \\\"ANC\\\")\"}
</answer>

Important:
- Use double quotes around phrases
- Use AND, OR, NOT operators (uppercase)
- Use parentheses to group terms logically
- Focus on product attributes, features, and descriptive terms'''

train_file = '$DATA_SRC/train/queries_product_train_step9.jsonl'
test_file = '$DATA_SRC/test/queries_product_test_step9_gt_v2.jsonl'
output_dir = '$WORK_DIR/data'

def load_queries(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f]

train_q = load_queries(train_file)
test_q = load_queries(test_file)

def build_records(queries, split):
    records = []
    for i, q in enumerate(queries):
        gt_ids = q.get('gt_document_ids') or []
        if not gt_ids: continue
        records.append({
            'data_source': f'ecommerce_{split}',
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.get('natural_language_query', '')},
            ],
            'ability': 'product_retrieval',
            'reward_model': json.dumps({'style': 'rule', 'ground_truth': {'target': gt_ids}}),
            'extra_info': json.dumps({'query_id': q.get('query_id',''), 'index': i, 'level': q.get('level'), 'modalities': q.get('modalities',[]), 'split': split}),
        })
    return records

train_ds = Dataset.from_list(build_records(train_q, 'train')).shuffle(seed=42)
val_ds = Dataset.from_list(build_records(test_q[:300], 'val'))

train_ds.to_parquet(f'{output_dir}/train.parquet')
val_ds.to_parquet(f'{output_dir}/val.parquet')
print(f'train: {len(train_ds)}, val: {len(val_ds)}')
"

echo "=== Step 6: Copy reward function ==="
# reward_deepretrieval.py should be placed at $WORK_DIR/reward_deepretrieval.py
# (scp from local or paste from repo)
if [ ! -f "$WORK_DIR/reward_deepretrieval.py" ]; then
    echo "WARNING: reward_deepretrieval.py not found at $WORK_DIR/"
    echo "Copy it: scp reward_ecommerce.py sunzhen@102:$WORK_DIR/reward_deepretrieval.py"
fi

echo ""
echo "=== Setup complete ==="
echo "To train: bash run_deepretrieval_grpo.sh"
