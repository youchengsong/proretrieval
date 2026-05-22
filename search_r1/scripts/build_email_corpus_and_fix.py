"""Build email corpus from email_test.json, then regenerate email SFT data."""
import json, os

BASE = "/root/autodl-tmp"

# Build email corpus
print("=== Building email corpus ===")
with open(f"{BASE}/data/email/email_test.json") as f:
    emails = json.load(f)

corpus_path = f"{BASE}/search_r1/corpus/email_corpus.jsonl"
with open(corpus_path, "w") as fout:
    for email in emails:
        doc = {
            "id": email["Message_ID"],
            "title": f"From: {email.get('From','')} Subject: {email.get('Subject','')}",
            "text": f"To: {email.get('To','')} Date: {email.get('Date','')} {email.get('Message','')[:500]}"
        }
        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"  Written {len(emails)} docs to {corpus_path}")

# Verify
with open(corpus_path) as f:
    first = json.loads(f.readline())
    print(f"  Sample: id={first['id'][:40]}, title={first['title'][:60]}")

print("\n=== Now regenerate email SFT data ===")
# Run the v2 fix script (it will use the updated corpus)
exec(open(f"{BASE}/scripts/fix_sft_data_v2.py").read())
