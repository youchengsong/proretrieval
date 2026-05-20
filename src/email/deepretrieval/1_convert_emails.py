#!/usr/bin/env python3
"""Convert email data to Pyserini-compatible JSONL format for BM25 indexing.

Supports both JSON (list) and Parquet input formats.
Email fields: Message_ID, From, To, Subject, Message
"""
import json
import os
import sys

BASE_DIR = "/root/autodl-tmp/DeepRetrieval_email"
DATA_DIR = "/root/autodl-tmp/data/email"

TRAIN_CORPUS = os.path.join(DATA_DIR, "train/email_train.parquet")
TEST_CORPUS = os.path.join(DATA_DIR, "test/email_test.json")

TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "jsonl_docs_train")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "jsonl_docs_test")


def load_emails(filepath):
    if filepath.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(filepath)
        return df.to_dict("records")
    else:
        with open(filepath) as f:
            return json.load(f)


def convert(emails, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "emails.jsonl")
    with open(output_file, "w", encoding="utf-8") as out:
        for e in emails:
            parts = []
            if e.get("From"):
                parts.append(f"From: {e['From']}")
            if e.get("To"):
                parts.append(f"To: {e['To']}")
            if e.get("Subject"):
                parts.append(f"Subject: {e['Subject']}")
            if e.get("Message"):
                parts.append(e["Message"])
            contents = " ".join(parts)
            doc = {"id": e["Message_ID"], "contents": contents}
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Wrote {len(emails)} documents to {output_file}")


def main():
    print("Converting train emails...")
    train_emails = load_emails(TRAIN_CORPUS)
    print(f"Loaded {len(train_emails)} train emails")
    convert(train_emails, TRAIN_OUTPUT_DIR)

    print("Converting test emails...")
    test_emails = load_emails(TEST_CORPUS)
    print(f"Loaded {len(test_emails)} test emails")
    convert(test_emails, TEST_OUTPUT_DIR)


if __name__ == "__main__":
    main()
