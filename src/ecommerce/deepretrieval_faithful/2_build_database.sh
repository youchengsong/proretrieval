#!/bin/bash
# Build Pyserini BM25 index for ecommerce products.
# Mirrors DeepRetrieval's src/Lucene/msmarco_beir/2_build_database.sh
set -e

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

INPUT_DIR=code/data/local_index_search/ecommerce/jsonl_docs
INDEX_DIR=code/data/local_index_search/ecommerce/pyserini_index

echo "Building Lucene BM25 index from $INPUT_DIR..."
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
