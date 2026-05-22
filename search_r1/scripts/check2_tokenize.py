"""CHECK 2: tokenize + label masking verification"""
import json, random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)

ASSISTANT_HEADER = '<|im_start|>assistant\n'
assistant_header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

def build_chat_text(messages):
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    return ''.join(parts)

def find_sublist(lst, sublist):
    for i in range(len(lst) - len(sublist) + 1):
        if lst[i:i+len(sublist)] == sublist:
            return i
    return -1

print('='*60)
print('CHECK 2: tokenize + label masking')
print('='*60)

datasets = [
    ('/root/autodl-tmp/search_r1/sft_data/ecommerce_search_sft.jsonl', 'search_ecom'),
    ('/root/autodl-tmp/search_r1/sft_data/email_search_sft.jsonl', 'search_email'),
    ('/root/autodl-tmp/sql_r1/sft_data/ecommerce_sql_sft.jsonl', 'sql_ecom'),
    ('/root/autodl-tmp/sql_r1/sft_data/email_sql_sft.jsonl', 'sql_email'),
]

for path, name in datasets:
    with open(path) as f:
        lines = f.readlines()

    indices = random.sample(range(len(lines)), 20)
    token_lengths = []
    trained_lengths = []
    all_ok = True

    for idx in indices:
        d = json.loads(lines[idx])
        text = build_chat_text(d['messages'])
        encoded = tokenizer(text, truncation=True, max_length=2048, padding=False)
        input_ids = encoded['input_ids']

        labels = [-100] * len(input_ids)
        pos = find_sublist(input_ids, assistant_header_ids)
        if pos < 0:
            print(f'  {name} sample {idx}: CANNOT FIND ASSISTANT HEADER!')
            all_ok = False
            continue
        start = pos + len(assistant_header_ids)
        labels[start:] = input_ids[start:]

        token_lengths.append(len(input_ids))
        trained_lengths.append(len(input_ids) - start)

        trained_text = tokenizer.decode([l for l in labels if l != -100])
        if 'search' in name and '<search>' not in trained_text:
            print(f'  {name} sample {idx}: MISSING <search> in trained text!')
            all_ok = False
        if '<answer>' not in trained_text:
            print(f'  {name} sample {idx}: MISSING <answer> in trained text!')
            all_ok = False

    avg_tokens = sum(token_lengths) / len(token_lengths)
    avg_trained = sum(trained_lengths) / len(trained_lengths)
    max_tokens = max(token_lengths)
    truncated = sum(1 for t in token_lengths if t >= 2048)

    status = 'PASS' if all_ok else 'FAIL'
    print(f'{name}: [{status}] avg={avg_tokens:.0f} tok (max={max_tokens}), trained={avg_trained:.0f} ({avg_trained/avg_tokens*100:.0f}%), truncated={truncated}/20')

print()
print('='*60)
print('CHECK 3: 训练脚本 train_sft_v2.py 关键逻辑')
print('='*60)

# Simulate what the training script does on one sample
with open('/root/autodl-tmp/search_r1/sft_data/ecommerce_search_sft.jsonl') as f:
    sample = json.loads(f.readline())

text = build_chat_text(sample['messages'])
encoded = tokenizer(text, truncation=True, max_length=2048, padding=False, return_tensors=None)
input_ids = encoded['input_ids']
labels = [-100] * len(input_ids)
idx = find_sublist(input_ids, assistant_header_ids)
start = idx + len(assistant_header_ids)
labels[start:] = input_ids[start:]

# Verify no -100 in trained region
trained_region = labels[start:]
neg100_in_trained = sum(1 for l in trained_region if l == -100)
print(f'Unexpected -100 in trained region: {neg100_in_trained} (should be 0)')

# Verify labels match input_ids where not masked
for i in range(len(labels)):
    if labels[i] != -100 and labels[i] != input_ids[i]:
        print(f'Label mismatch at position {i}: label={labels[i]}, input_id={input_ids[i]}')
        break
else:
    print('Label-input_id consistency: PASS')

# Verify assistant header tokens
print(f'Assistant header IDs: {assistant_header_ids}')
print(f'Found at position: {idx}')
print(f'Tokens at that position: {input_ids[idx:idx+len(assistant_header_ids)]}')
print(f'Match: {input_ids[idx:idx+len(assistant_header_ids)] == assistant_header_ids}')

# Verify DataCollatorForSeq2Seq will pad labels with -100
print(f'Pad token ID: {tokenizer.pad_token_id}')
print(f'EOS token ID: {tokenizer.eos_token_id}')
if tokenizer.pad_token_id is None:
    print('WARNING: pad_token is None - script sets it to eos_token')

print()
print('='*60)
print('CHECK 4: 推理时的模板一致性')
print('='*60)

# Training template (full conversation)
train_text = build_chat_text(sample['messages'])

# Inference template (system + user, add assistant header)
inf_msgs = sample['messages'][:-1]  # system + user only
inf_parts = []
for m in inf_msgs:
    inf_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
inf_parts.append("<|im_start|>assistant\n")
inf_text = "".join(inf_parts)

# The inference text should be a PREFIX of the training text
if train_text.startswith(inf_text):
    print('Inference template is prefix of training template: PASS')
else:
    print('Inference template is NOT prefix of training template: FAIL')
    print(f'Training starts with: {repr(train_text[:200])}')
    print(f'Inference text:       {repr(inf_text[:200])}')

print()
print('='*60)
print('CHECK 5: SQL数据格式验证')
print('='*60)

with open('/root/autodl-tmp/sql_r1/sft_data/ecommerce_sql_sft.jsonl') as f:
    sql_sample = json.loads(f.readline())

sql_asst = sql_sample['messages'][-1]['content']
print(f'SQL assistant content (first 300 chars):')
print(sql_asst[:300])
print()

sql_text = build_chat_text(sql_sample['messages'])
sql_enc = tokenizer(sql_text, truncation=True, max_length=2048, padding=False)
sql_ids = sql_enc['input_ids']
sql_pos = find_sublist(sql_ids, assistant_header_ids)
sql_start = sql_pos + len(assistant_header_ids)
sql_trained = tokenizer.decode(sql_ids[sql_start:])
print(f'SQL trained text (first 300 chars):')
print(sql_trained[:300])
has_sql = '```sql' in sql_trained or 'SELECT' in sql_trained
print(f'Has SQL: {has_sql}')
