"""
Qwen3-1.7B SFT 训练
- DeepSpeed ZeRO-2
- 2x A100 80GB
- 复用 4B 版本的训练数据
"""

import os, json, random
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTTrainer, SFTConfig

# ── 路径配置 ──
MODEL_NAME = "/data/LLM_Models/Qwen3-1.7B"
DATA_PATH  = "/home/sunzhen/LocalSearch/ecommerce/train.jsonl"
OUTPUT_DIR = "/data/s3-migration/sft_localSearch_1.7b/output"
DS_CONFIG  = "/home/sunzhen/LocalSearch/ecommerce_1.7b/ds_zero2.json"
SEED       = 42

# ── 随机种子 ──
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# ── Flash Attention ──
try:
    import flash_attn
    attn_impl = "flash_attention_2"
    print("Using Flash Attention 2")
except ImportError:
    attn_impl = "eager"
    print("Using eager attention")

# ── 加载数据 ──
print("Loading data...")
records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))
dataset = Dataset.from_list(records)
print(f"Total samples: {len(dataset)}")

# ── Tokenizer ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_messages(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_messages, num_proc=4, remove_columns=["messages"])

# ── 断点续传 ──
resume_from_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = sorted(
        [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if ckpts:
        resume_from_checkpoint = os.path.join(OUTPUT_DIR, ckpts[-1])
        print(f"Resuming from: {resume_from_checkpoint}")

# ── 训练参数 ──
# 2 GPU x micro_batch 2 x grad_accum 8 = effective batch 32
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    data_seed=SEED,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
    max_length=2048,
    dataset_text_field="text",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    logging_steps=10,
    logging_dir=OUTPUT_DIR + "/logs",
    report_to="none",
    # deepspeed=DS_CONFIG,  # 单卡不需要
    dataloader_num_workers=4,
)

# ── 模型 ──
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl, trust_remote_code=True,
)

# ── 训练 ──
trainer = SFTTrainer(
    model=model, args=training_args,
    train_dataset=dataset, processing_class=tokenizer,
)
print("Start training...")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

final_dir = OUTPUT_DIR + "/final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Done. Model saved to {final_dir}")
