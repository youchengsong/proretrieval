"""SFT training using transformers.Trainer directly (no TRL dependency)."""
import os, sys, json, random, argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    set_seed, DataCollatorForSeq2Seq
)

SEED = 42

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

set_all_seeds(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        print("Using SDPA attention")

    print(f"Loading data from {args.data_path}...")
    records = []
    with open(args.data_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Total samples: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ASSISTANT_HEADER = "<|im_start|>assistant\n"
    assistant_header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    def build_chat_text(messages):
        """Build Qwen3 chat format manually to avoid template stripping tags."""
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        return "".join(parts)

    def find_sublist(lst, sublist):
        for i in range(len(lst) - len(sublist) + 1):
            if lst[i:i+len(sublist)] == sublist:
                return i
        return -1

    def tokenize_messages(example):
        text = build_chat_text(example["messages"])
        encoded = tokenizer(
            text, truncation=True, max_length=args.max_length,
            padding=False, return_tensors=None
        )
        input_ids = encoded["input_ids"]
        labels = [-100] * len(input_ids)

        idx = find_sublist(input_ids, assistant_header_ids)
        if idx >= 0:
            start = idx + len(assistant_header_ids)
            labels[start:] = input_ids[start:]

        encoded["labels"] = labels
        return encoded

    dataset = Dataset.from_list(records)
    dataset = dataset.map(tokenize_messages, remove_columns=dataset.column_names, num_proc=4)
    print(f"Tokenized samples: {len(dataset)}")

    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    ds_config_path = os.path.join(args.output_dir, "ds_zero2.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=10,
        logging_dir=args.output_dir + "/logs",
        report_to="none",
        deepspeed=ds_config_path,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    final_dir = args.output_dir + "/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"SFT Done. Model saved to {final_dir}")

if __name__ == "__main__":
    main()
