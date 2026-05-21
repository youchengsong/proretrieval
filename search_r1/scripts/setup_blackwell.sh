#!/bin/bash
set -e

echo "=== Step 1: Convert ecommerce corpus to JSONL ==="
python3 /root/autodl-tmp/scripts/convert_corpus_to_jsonl.py ecommerce \
    /root/autodl-tmp/data/ecommerce/products_train.parquet \
    /root/autodl-tmp/data/ecommerce/bm25_corpus

echo "=== Step 2: Build BM25 index for ecommerce ==="
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input /root/autodl-tmp/data/ecommerce/bm25_corpus/ \
    --index /root/autodl-tmp/data/ecommerce/bm25_index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions --storeDocvectors --storeRaw

echo "=== Step 3: Verify BM25 index ==="
python3 -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('/root/autodl-tmp/data/ecommerce/bm25_index')
print(f'Index contains {s.num_docs} documents')
hits = s.search('red shoes', k=3)
for h in hits:
    print(f'  {h.docid}: score={h.score:.4f}')
print('BM25 index OK')
"

echo "=== Step 4: Convert email corpus to JSONL ==="
python3 /root/autodl-tmp/scripts/convert_corpus_to_jsonl.py email \
    /root/autodl-tmp/data/email/email_train.parquet \
    /root/autodl-tmp/data/email/bm25_corpus

echo "=== Step 5: Build BM25 index for email ==="
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input /root/autodl-tmp/data/email/bm25_corpus/ \
    --index /root/autodl-tmp/data/email/bm25_index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions --storeDocvectors --storeRaw

echo "=== Step 6: Verify email BM25 index ==="
python3 -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('/root/autodl-tmp/data/email/bm25_index')
print(f'Email index contains {s.num_docs} documents')
hits = s.search('budget meeting', k=3)
for h in hits:
    print(f'  {h.docid}: score={h.score:.4f}')
print('Email BM25 index OK')
"

echo "=== Step 7: Install Search-R1 modified verl ==="
cd /root/autodl-tmp/Search-R1
pip install -e . 2>&1 | tail -5

echo "=== Step 8: Verify modified verl ==="
python3 -c "
try:
    from search_r1.llm_agent.generation import LLMGenerationManager
    print('LLMGenerationManager import OK')
except Exception as e:
    print(f'Import failed: {e}')
"

echo "=== ALL SETUP DONE ==="
