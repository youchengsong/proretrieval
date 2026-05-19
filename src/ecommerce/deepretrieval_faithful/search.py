"""Pyserini BM25 search module for ecommerce products.

Direct copy of DeepRetrieval's src/Lucene/msmarco_beir/search.py
with only the default index_dir changed to our ecommerce index.
"""
import json
from pyserini.search.lucene import LuceneSearcher


class PyseriniMultiFieldSearch:
    def __init__(self, index_dir="code/data/local_index_search/ecommerce/pyserini_index"):
        self.searcher = LuceneSearcher(index_dir)

    def batch_search(self, queries, top_k=10, threads=4):
        field_queries = [
            f"contents:{query}"
            for query in queries
        ]

        results_dict = self.searcher.batch_search(
            field_queries,
            [str(i) for i in range(len(queries))],
            k=top_k,
            threads=threads
        )

        final_results = {}
        for i, query in enumerate(queries):
            hits = results_dict[str(i)]
            formatted_results = [
                (json.loads(hit.raw)["id"], json.loads(hit.raw)["contents"], hit.score)
                for hit in hits
            ]
            final_results[query] = formatted_results

        return final_results


if __name__ == "__main__":
    search_system = PyseriniMultiFieldSearch()
    queries = ["running shoes"]
    results = search_system.batch_search(queries, top_k=5, threads=4)
    for query, items in results.items():
        print(f"\nQuery: {query}")
        for doc_id, text, score in items:
            print(f"  ID: {doc_id}, Score: {score:.4f}, Text: {text[:100]}...")
