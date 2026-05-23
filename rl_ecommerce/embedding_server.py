"""
Embedding server: loads text + image encoder on a dedicated GPU,
serves encoding requests via HTTP for reward workers.

Usage:
    EMBED_GPU=5 python embedding_server.py
"""
import os, sys, json, signal, threading, logging, time
import numpy as np
from flask import Flask, request, jsonify

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EMBED_GPU", "2")

import torch

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__)
_gpu_lock = threading.Lock()

DEVICE = "cuda:0"
TEXT_MODEL_PATH = os.environ.get("TEXT_MODEL_PATH", "/data/youchengsong/models/Qwen3-Embedding-8B")
IMAGE_MODEL_PATH = os.environ.get("IMAGE_MODEL_PATH", "/data/youchengsong/models/Qwen3-VL-Embedding-8B")

_text_encoder = None
_image_encoder = None
_image_processor = None

_stats = {"text_ok": 0, "text_err": 0, "image_ok": 0, "image_err": 0, "start": 0}
_stats_lock = threading.Lock()
_STATS_INTERVAL = 100


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[embed_server {ts}] {msg}", flush=True)


def _stats_bump(key):
    with _stats_lock:
        _stats[key] += 1
        total = _stats["text_ok"] + _stats["text_err"] + _stats["image_ok"] + _stats["image_err"]
        if total % _STATS_INTERVAL == 0:
            elapsed = time.time() - _stats["start"]
            _log(f"requests={total} | text={_stats['text_ok']}ok/{_stats['text_err']}err "
                 f"image={_stats['image_ok']}ok/{_stats['image_err']}err | {elapsed:.0f}s")


def _load_models():
    global _text_encoder, _image_encoder, _image_processor
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor

    _log(f"Loading text encoder (fp16) on {DEVICE}...")
    _text_encoder = SentenceTransformer(
        TEXT_MODEL_PATH, device=DEVICE,
        model_kwargs={"torch_dtype": torch.float16}
    )
    _text_encoder.max_seq_length = 128
    _log("Text encoder ready.")

    _log(f"Loading image encoder (fp16) on {DEVICE}...")
    _image_encoder = AutoModel.from_pretrained(
        IMAGE_MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    _image_processor = AutoProcessor.from_pretrained(IMAGE_MODEL_PATH)
    _log("Image encoder ready.")

    mem = torch.cuda.memory_allocated() / 1024**3
    _log(f"GPU memory used: {mem:.1f} GB")
    _stats["start"] = time.time()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/encode_text", methods=["POST"])
def encode_text():
    try:
        queries = request.json["queries"]
        if not queries:
            return jsonify({"vectors": {}})
        with _gpu_lock:
            vecs = _text_encoder.encode(
                queries, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
        result = {q: v.tolist() for q, v in zip(queries, vecs)}
        _stats_bump("text_ok")
        return jsonify({"vectors": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        _stats_bump("text_err")
        return jsonify({"error": str(e), "vectors": {}}), 500


@app.route("/encode_image", methods=["POST"])
def encode_image():
    try:
        queries = request.json["queries"]
        if not queries:
            return jsonify({"vectors": {}})
        import torch.nn.functional as F

        all_embs = {}
        with _gpu_lock:
            for i in range(0, len(queries), 32):
                batch = queries[i:i+32]
                msgs = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in batch]
                texts = [_image_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
                inputs = _image_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                with torch.no_grad():
                    out = _image_encoder(**inputs, output_hidden_states=True)
                    hidden = out.hidden_states[-1]
                    seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                    embs = hidden[torch.arange(len(batch), device=DEVICE), seq_lens]
                    embs = F.normalize(embs, p=2, dim=-1).float().cpu().numpy()
                for j, q in enumerate(batch):
                    all_embs[q] = embs[j].tolist()

        _stats_bump("image_ok")
        return jsonify({"vectors": all_embs})
    except Exception as e:
        import traceback
        traceback.print_exc()
        _stats_bump("image_err")
        return jsonify({"error": str(e), "vectors": {}}), 500


if __name__ == "__main__":
    _load_models()
    port = int(os.environ.get("EMBED_PORT", "9123"))
    _log(f"Serving on port {port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
