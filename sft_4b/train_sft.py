"""
Qwen3-4B 全量 SFT
- DeepSpeed ZeRO-2（比 ZeRO-3 通信少，速度更快）
- Flash Attention 2（如已安装）
- 固定随机种子（模型/数据/torch/cuda 全覆盖）
- 断点续传（自动检测 output_dir 中最新 checkpoint）
"""

import os
import json
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

# ── 路径配置 ────────────────────────────────────────────────
MODEL_NAME = "/data/LLM_Models/Qwen3-4B"
DATA_PATH  = "/home/sunzhen/sft_localSearch/train.jsonl"
OUTPUT_DIR = "/data/s3-migration/sft_localSearch/output"
DS_CONFIG  = "/home/sunzhen/sft_localSearch/ds_zero2.json"
SEED       = 42

# ── 随机种子（断点续传时必须与初次一致）──────────────────────
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# ── 检测 Flash Attention 2 ─────────────────────────────────
try:
    import flash_attn
    attn_impl = "flash_attention_2"
    print("Flash Attention 2 available, using it.")
except ImportError:
    attn_impl = "eager"
    print("Flash Attention 2 not available, using eager attention.")

# ── 加载数据 ────────────────────────────────────────────────
print("Loading data...")
records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))
dataset = Dataset.from_list(records)
print(f"Total samples: {len(dataset)}")

# ── Tokenizer ───────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 格式化：messages -> text ────────────────────────────────
def format_messages(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_messages, num_proc=4, remove_columns=["messages"])

# ── 自动检测 checkpoint 以续传 ──────────────────────────────
resume_from_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    ckpts = sorted(
        [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if ckpts:
        resume_from_checkpoint = os.path.join(OUTPUT_DIR, ckpts[-1])
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        print("No checkpoint found, training from scratch.")
else:
    print("Output dir not found, training from scratch.")

# ── 训练参数 ────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    data_seed=SEED,

    # 轮次
    num_train_epochs=3,

    # batch：4 GPU × micro_batch 2 × grad_accum 4 = effective batch 32
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # 优化器
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_torch",

    # 精度（不开 gradient_checkpointing，A6000 48GB 够用，速度更快）
    bf16=True,
    gradient_checkpointing=True,

    # 序列长度
    max_length=2048,
    dataset_text_field="text",

    # 保存：每 50 步存一次，最多保留 3 个
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,

    # 日志
    logging_steps=10,
    logging_dir=OUTPUT_DIR + "/logs",
    report_to="none",

    # DeepSpeed ZeRO-2
    deepspeed=DS_CONFIG,

    # dataloader
    dataloader_num_workers=4,
)

# ── 模型 ────────────────────────────────────────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)

# ── Trainer ─────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ── 训练（自动续传）─────────────────────────────────────────
print("Start training...")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# ── 保存最终模型 ─────────────────────────────────────────────
final_dir = OUTPUT_DIR + "/final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Done. Model saved to {final_dir}")
