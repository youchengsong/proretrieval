#!/bin/bash
# Build Pyserini BM25 indexes for email corpus (train + test)
set -e

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=/root/miniconda3/bin:$PATH

BASE_DIR="/root/autodl-tmp/DeepRetrieval_email"

build_index() {
    local input_dir="$1"
    local index_dir="$2"
    local name="$3"

    echo "Building Lucene BM25 index for ${name}..."
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input "$input_dir" \
        --index "$index_dir" \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4 \
        --storePositions \
        --storeDocvectors \
        --storeRaw

    echo "Verifying ${name} index..."
    python -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher('${index_dir}')
print(f'${name} index contains {searcher.num_docs} documents')
hits = searcher.search('meeting schedule', k=3)
for h in hits:
    print(f'  {h.docid}: {h.score:.4f}')
print('${name} index verification complete')
"
}

build_index "${BASE_DIR}/jsonl_docs_train" "${BASE_DIR}/indexes/lucene-index-emails-train" "train"
build_index "${BASE_DIR}/jsonl_docs_test" "${BASE_DIR}/indexes/lucene-index-emails-test" "test"

echo "All indexes built successfully."
