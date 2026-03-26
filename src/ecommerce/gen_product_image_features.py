#!/usr/bin/env python3
"""
用 Qwen3-VL-Embedding-8B 提取电商图片特征
输入：products_train.parquet / products_test.parquet
输出：product_image_features_train.parquet / product_image_features_test.parquet
       列：product_id (str), image_features (np.float32 array)

运行：
    python extract_image_features.py
内部自动拉起 4 个子进程，各占一张 GPU（2,4,5,6），最后合并。
"""

import argparse, json, os, subprocess, sys
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/data/youchengsong/models/Qwen3-VL-Embedding-8B"
IMAGE_DIR   = Path("/data/youchengsong/DSL-R1/image")
DATA_DIR    = Path("/data/youchengsong/DSL-R1/data/ecommerce")
PYTHON      = "/data/youchengsong/conda3/envs/DSL-R1/bin/python"
GPUS        = [2, 4, 5, 6]
BATCH_SIZE  = 8

SPLITS = {
    "train": DATA_DIR / "products_train.parquet",
    "test":  DATA_DIR / "products_test.parquet",
}
OUTPUTS = {
    "train": DATA_DIR / "product_image_features_train.parquet",
    "test":  DATA_DIR / "product_image_features_test.parquet",
}


# ── shard worker ──────────────────────────────────────────────────────────────
def run_shard(split: str, shard: int, total: int):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{split} shard {shard}/{total}] device={device}", flush=True)

    # 加载模型
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=str(device),
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 加载当前 shard 的数据
    df = pd.read_parquet(SPLITS[split])
    df_shard = df.iloc[shard::total].reset_index(drop=True)
    print(f"[{split} shard {shard}/{total}] {len(df_shard)} products", flush=True)

    def encode_images(product_ids):
        images, valid_ids = [], []
        for pid in product_ids:
            img_path = IMAGE_DIR / f"{pid}.jpg"
            if img_path.exists():
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    valid_ids.append(pid)
                except Exception as e:
                    print(f"  [warn] {pid}: {e}", flush=True)
            else:
                print(f"  [warn] 图片不存在: {img_path}", flush=True)
        if not images:
            return [], []

        # 构造输入：纯图像，无文本指令（document encoding）
        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}]}]
            for img in images
        ]
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_batch
        ]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]          # (B, L, D)
            # 取每条序列最后一个有效 token（EOS）的隐层作为 embedding
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
            embs = hidden[torch.arange(len(images), device=device), seq_lens]  # (B, D)
            embs = F.normalize(embs, p=2, dim=-1).float().cpu().numpy()

        return valid_ids, embs

    all_ids, all_feats = [], []
    pids = df_shard["product_id"].tolist()
    for i in range(0, len(pids), BATCH_SIZE):
        batch = pids[i: i + BATCH_SIZE]
        ids, feats = encode_images(batch)
        all_ids.extend(ids)
        all_feats.extend(feats)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  [{split} shard {shard}] {i}/{len(pids)}", flush=True)

    # 保存 shard 结果
    out_path = DATA_DIR / f"_img_shard_{split}_{shard}.parquet"
    result_df = pd.DataFrame({
        "product_id":     all_ids,
        "image_features": [f for f in all_feats],
    })
    result_df.to_parquet(out_path, index=False)
    print(f"[{split} shard {shard}] done → {out_path} ({len(result_df)} rows)", flush=True)


# ── 主进程：拉起子进程 + 合并 ─────────────────────────────────────────────────
def main():
    total = len(GPUS)

    for split in ["train", "test"]:
        print(f"\n=== {split} ({SPLITS[split].name}) ===")
        print(f"拉起 {total} 个子进程，GPU={GPUS}")

        procs = []
        for shard, gpu in enumerate(GPUS):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            p = subprocess.Popen(
                [PYTHON, __file__, "--_shard", str(shard), "--_total", str(total), "--_split", split],
                env=env,
            )
            procs.append(p)
            print(f"  shard {shard} → GPU {gpu} (pid={p.pid})")

        for p in procs:
            p.wait()
        print(f"所有 shard 完成，开始合并...")

        # 合并
        import pandas as pd
        import numpy as np
        dfs = []
        for shard in range(total):
            p = DATA_DIR / f"_img_shard_{split}_{shard}.parquet"
            dfs.append(pd.read_parquet(p))
            p.unlink()

        merged = pd.concat(dfs, ignore_index=True)
        # 恢复原始顺序
        orig_df = pd.read_parquet(SPLITS[split])
        merged = orig_df[["product_id"]].merge(merged, on="product_id", how="left")

        merged.to_parquet(OUTPUTS[split], index=False)
        print(f"合并完成 → {OUTPUTS[split]}  ({len(merged)} 行)")
        missing = merged["image_features"].isna().sum()
        if missing:
            print(f"  [warn] 缺失特征: {missing} 行")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_shard", type=int, default=None)
    parser.add_argument("--_total", type=int, default=None)
    parser.add_argument("--_split", type=str, default=None)
    args = parser.parse_args()

    if args._shard is not None:
        run_shard(args._split, args._shard, args._total)
    else:
        main()
