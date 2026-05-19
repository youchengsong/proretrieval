#!/bin/bash
# Build Pyserini BM25 index from product JSONL
set -e

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=/home/sunzhen/miniconda3/envs/mlx/bin:$PATH

INPUT_DIR="/home/sunzhen/DeepRetrieval_ecommerce/jsonl_docs"
INDEX_DIR="/home/sunzhen/DeepRetrieval_ecommerce/indexes/lucene-index-products"

echo "Building Lucene BM25 index..."
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$INPUT_DIR" \
    --index "$INDEX_DIR" \
    --generator DefaultLuceneDocumentGenerator \
    --threads 4 \
    --storePositions \
    --storeDocvectors \
    --storeRaw

echo "Index built at $INDEX_DIR"
echo "Verifying index..."
python -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('$INDEX_DIR')
print(f'Index contains {searcher.num_docs} documents')
hits = searcher.search('running shoes', k=3)
for h in hits:
    print(f'  {h.docid}: {h.score:.4f}')
print('Index verification complete')
"
