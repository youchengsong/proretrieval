#!/usr/bin/env python3
"""
人工审查工具：逐条审查 DSL + 自然语言 query，支持断点续审。

用法：
    python review_nl_query.py --input /path/to/step9.jsonl --output /path/to/review.jsonl
    python review_nl_query.py --input /path/to/step9.jsonl --output /path/to/review.jsonl --port 8080

审查结果保存到 output 文件，每条记录格式：
    {"query_id": "q_00001", "action": "approve|reject|edit", "nl_query": "..."}

快捷键：
    A / Enter  → 通过
    R          → 拒绝
    E          → 编辑后保存
    ← →        → 上一条 / 下一条（仅浏览，不记录）
"""

import argparse
import json
import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
queries   = []
results   = {}   # query_id -> {action, nl_query}
INPUT_FILE  = ""
OUTPUT_FILE = ""

def load_queries(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def load_results(path):
    r = {}
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                r[obj["query_id"]] = obj
    return r

def save_result(obj):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/queries/count")
def count():
    reviewed = len(results)
    return jsonify({"total": len(queries), "reviewed": reviewed})

@app.route("/api/queries/<int:idx>")
def get_query(idx):
    if idx < 0 or idx >= len(queries):
        return jsonify({"error": "out of range"}), 404
    q = queries[idx]
    r = results.get(q["query_id"])
    return jsonify({
        "idx":       idx,
        "total":     len(queries),
        "query_id":  q["query_id"],
        "level":     q.get("level"),
        "n":         q.get("n"),
        "gt_valid":  q.get("gt_document_ids") is not None,
        "gt_size":   len(q["gt_document_ids"]) if q.get("gt_document_ids") else 0,
        "dsl":       q.get("dsl"),
        "leaves":    q.get("leaves"),
        "groups":    q.get("groups"),
        "inner_op":  q.get("inner_op"),
        "outer_op":  q.get("outer_op"),
        "nl_query":  q.get("natural_language_query"),
        "reviewed":  r,
    })

@app.route("/api/next-unreviewed/<int:from_idx>")
def next_unreviewed(from_idx):
    for i in range(from_idx, len(queries)):
        if queries[i]["query_id"] not in results:
            return jsonify({"idx": i})
    return jsonify({"idx": -1})

@app.route("/api/review", methods=["POST"])
def review():
    data     = request.json
    query_id = data["query_id"]
    action   = data["action"]   # approve / reject / edit
    nl_query = data.get("nl_query", "")

    obj = {"query_id": query_id, "action": action, "nl_query": nl_query}

    if query_id not in results:
        save_result(obj)
    results[query_id] = obj

    return jsonify({"ok": True})

@app.route("/api/stats")
def stats():
    approve = sum(1 for r in results.values() if r["action"] == "approve")
    reject  = sum(1 for r in results.values() if r["action"] == "reject")
    edit    = sum(1 for r in results.values() if r["action"] == "edit")
    return jsonify({
        "total":    len(queries),
        "reviewed": len(results),
        "approve":  approve,
        "reject":   reject,
        "edit":     edit,
    })

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>NL Query 人工审查</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #222; }

#header {
  background: #1a1a2e; color: #eee; padding: 12px 24px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
}
#header h1 { font-size: 18px; font-weight: 600; }
#progress-bar-wrap { flex: 1; margin: 0 24px; background: #333; border-radius: 4px; height: 8px; }
#progress-bar { height: 8px; background: #4ade80; border-radius: 4px; transition: width 0.3s; }
#progress-text { font-size: 13px; white-space: nowrap; }

#nav {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 24px; background: #fff; border-bottom: 1px solid #ddd;
}
#nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid #ccc;
  cursor: pointer; font-size: 13px; background: #fff; }
#nav button:hover { background: #f5f5f5; }
#idx-input { width: 70px; padding: 5px 8px; border: 1px solid #ccc;
  border-radius: 6px; font-size: 13px; text-align: center; }
#meta { font-size: 13px; color: #666; margin-left: 8px; }

#main { display: flex; gap: 16px; padding: 16px 24px; max-width: 1600px; margin: 0 auto; }

.panel { background: #fff; border-radius: 10px; padding: 18px; flex: 1;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08); overflow: auto; }
.panel h2 { font-size: 14px; font-weight: 700; color: #555;
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; }

/* DSL panel */
#sql { font-family: monospace; font-size: 13px; background: #1e1e2e; color: #cdd6f4;
  padding: 12px 14px; border-radius: 8px; white-space: pre-wrap; word-break: break-all;
  margin-bottom: 14px; line-height: 1.6; }

.retrieval-item { background: #f8f9ff; border-left: 3px solid #6366f1;
  padding: 8px 12px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 13px; }
.retrieval-item .type { font-weight: 700; color: #6366f1; margin-right: 6px; }
.retrieval-item .query-text { color: #333; }

.leaf { padding: 7px 12px; margin-bottom: 6px; border-radius: 6px; font-size: 13px; }
.leaf.pos { background: #f0fdf4; border-left: 3px solid #22c55e; }
.leaf.neg { background: #fff1f2; border-left: 3px solid #f43f5e; }
.leaf .leaf-meta { color: #888; font-size: 11px; margin-bottom: 2px; }
.leaf .leaf-val { font-family: monospace; color: #222; }

/* NL panel */
#nl-current { font-size: 16px; line-height: 1.7; color: #111; background: #fafafa;
  border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
#nl-edit { width: 100%; height: 100px; font-size: 14px; padding: 10px 12px;
  border: 1px solid #d1d5db; border-radius: 8px; resize: vertical;
  display: none; font-family: inherit; line-height: 1.6; }

#actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
.btn { padding: 10px 24px; border: none; border-radius: 8px; font-size: 15px;
  font-weight: 600; cursor: pointer; transition: all 0.15s; }
.btn:active { transform: scale(0.96); }
.btn-approve { background: #22c55e; color: #fff; }
.btn-approve:hover { background: #16a34a; }
.btn-reject  { background: #f43f5e; color: #fff; }
.btn-reject:hover  { background: #e11d48; }
.btn-edit    { background: #f59e0b; color: #fff; }
.btn-edit:hover    { background: #d97706; }
.btn-save    { background: #6366f1; color: #fff; display: none; }
.btn-save:hover    { background: #4f46e5; }

.reviewed-badge { display: inline-block; padding: 4px 10px; border-radius: 20px;
  font-size: 12px; font-weight: 700; margin-bottom: 10px; }
.badge-approve { background: #dcfce7; color: #15803d; }
.badge-reject  { background: #ffe4e6; color: #be123c; }
.badge-edit    { background: #fef3c7; color: #92400e; }

#keyboard-hint { font-size: 12px; color: #999; margin-top: 14px; }
kbd { background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 4px;
  padding: 1px 5px; font-size: 11px; }
</style>
</head>
<body>

<div id="header">
  <h1>NL Query 人工审查</h1>
  <div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
  <div id="progress-text">0 / 0</div>
</div>

<div id="nav">
  <button onclick="go(currentIdx-1)">← 上一条</button>
  <input id="idx-input" type="number" min="0" value="0" onchange="go(parseInt(this.value))">
  <button onclick="go(currentIdx+1)">下一条 →</button>
  <button onclick="goNextUnreviewed()">跳到未审查 ⏩</button>
  <span id="meta"></span>
</div>

<div id="main">
  <!-- DSL Panel -->
  <div class="panel" style="flex:1.2">
    <h2>DSL</h2>
    <div id="sql"></div>

    <h2 style="margin-bottom:10px">Retrieval List</h2>
    <div id="retrieval-list"></div>

    <h2 style="margin-top:14px;margin-bottom:10px">Leaves</h2>
    <div id="leaves"></div>
  </div>

  <!-- NL Panel -->
  <div class="panel" style="flex:0.8">
    <h2>自然语言 Query</h2>
    <div id="reviewed-badge"></div>
    <div id="nl-current"></div>
    <textarea id="nl-edit" placeholder="编辑自然语言 query..."></textarea>

    <div id="actions">
      <button class="btn btn-approve" onclick="doAction('approve')">✓ 通过 (A)</button>
      <button class="btn btn-reject"  onclick="doAction('reject')">✗ 拒绝 (R)</button>
      <button class="btn btn-edit"    onclick="startEdit()">✎ 编辑 (E)</button>
      <button class="btn btn-save" id="btn-save" onclick="saveEdit()">💾 保存编辑</button>
    </div>

    <div id="keyboard-hint">
      快捷键：<kbd>A</kbd> 通过 &nbsp; <kbd>R</kbd> 拒绝 &nbsp;
      <kbd>E</kbd> 编辑 &nbsp; <kbd>←</kbd><kbd>→</kbd> 翻页
    </div>

    <div style="margin-top:20px;padding-top:14px;border-top:1px solid #eee">
      <h2>审查统计</h2>
      <div id="stats" style="font-size:13px;margin-top:8px;line-height:2"></div>
    </div>
  </div>
</div>

<script>
let currentIdx = 0;
let currentQuery = null;

async function go(idx) {
  if (idx < 0) idx = 0;
  const res = await fetch(`/api/queries/${idx}`);
  if (!res.ok) return;
  const q = await res.json();
  currentIdx = idx;
  currentQuery = q;
  render(q);
  document.getElementById('idx-input').value = idx;
  loadStats();
}

async function goNextUnreviewed() {
  const res = await fetch(`/api/next-unreviewed/${currentIdx}`);
  const data = await res.json();
  if (data.idx >= 0) go(data.idx);
  else alert('所有 query 已审查完毕！');
}

function render(q) {
  // SQL
  document.getElementById('sql').textContent = q.dsl?.sql || '';

  // Retrieval list
  const rl = q.dsl?.retrieval_list || [];
  document.getElementById('retrieval-list').innerHTML = rl.length === 0
    ? '<span style="color:#999;font-size:13px">（无）</span>'
    : rl.map((r,i) =>
        `<div class="retrieval-item">
          <span class="type">&lt;${r.type}_${i}&gt; [${r.field}]</span>
          <span class="query-text">${esc(r.query)}</span>
        </div>`).join('');

  // Leaves
  const leaves = q.leaves || [];
  document.getElementById('leaves').innerHTML = leaves.map(lf => {
    const cls = lf.is_pos ? 'pos' : 'neg';
    const pol = lf.is_pos ? '✓ pos' : '✗ neg';
    let val = '';
    if (lf.modality === 'text' || lf.modality === 'image') {
      val = lf.value?.val || '';
    } else {
      const v = lf.value || {};
      val = v.lo !== undefined ? `${v.lo} ~ ${v.hi}` : `${v.op} ${v.val}`;
    }
    return `<div class="leaf ${cls}">
      <div class="leaf-meta">leaf${lf.leaf_id} | ${lf.field} | ${lf.modality} | ${pol}</div>
      <div class="leaf-val">${esc(String(val))}</div>
    </div>`;
  }).join('');

  // Meta
  document.getElementById('meta').textContent =
    `#${currentIdx} | level=${q.level} n=${q.n} | GT: ${q.gt_valid ? q.gt_size+'条' : '空'}`;

  // NL
  const nlEl = document.getElementById('nl-current');
  nlEl.textContent = q.nl_query || '（无）';
  document.getElementById('nl-edit').value = q.nl_query || '';
  document.getElementById('nl-edit').style.display = 'none';
  document.getElementById('btn-save').style.display = 'none';

  // Badge
  const badge = document.getElementById('reviewed-badge');
  if (q.reviewed) {
    const a = q.reviewed.action;
    const cls = a === 'approve' ? 'badge-approve' : a === 'reject' ? 'badge-reject' : 'badge-edit';
    const label = a === 'approve' ? '✓ 已通过' : a === 'reject' ? '✗ 已拒绝' : '✎ 已编辑';
    badge.innerHTML = `<span class="reviewed-badge ${cls}">${label}</span>`;
    if (a === 'edit') nlEl.textContent = q.reviewed.nl_query || q.nl_query || '';
  } else {
    badge.innerHTML = '';
  }

  // Progress
  updateProgress(q.total);
}

async function doAction(action) {
  if (!currentQuery) return;
  const nl = document.getElementById('nl-current').textContent;
  await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query_id: currentQuery.query_id, action, nl_query: nl})
  });
  go(currentIdx + 1);
}

function startEdit() {
  const edit = document.getElementById('nl-edit');
  edit.style.display = 'block';
  document.getElementById('btn-save').style.display = 'inline-block';
  edit.focus();
}

async function saveEdit() {
  if (!currentQuery) return;
  const nl = document.getElementById('nl-edit').value.trim();
  await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query_id: currentQuery.query_id, action: 'edit', nl_query: nl})
  });
  go(currentIdx + 1);
}

async function loadStats() {
  const res = await fetch('/api/stats');
  const s = await res.json();
  document.getElementById('stats').innerHTML =
    `总数: <b>${s.total}</b> &nbsp;|&nbsp;
     已审: <b>${s.reviewed}</b> &nbsp;|&nbsp;
     通过: <b style="color:#16a34a">${s.approve}</b> &nbsp;
     拒绝: <b style="color:#e11d48">${s.reject}</b> &nbsp;
     编辑: <b style="color:#d97706">${s.edit}</b>`;
  updateProgress(s.total, s.reviewed);
}

function updateProgress(total, reviewed) {
  if (reviewed === undefined) return;
  const pct = total ? (reviewed / total * 100).toFixed(1) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${reviewed} / ${total} (${pct}%)`;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'a' || e.key === 'A' || e.key === 'Enter') doAction('approve');
  else if (e.key === 'r' || e.key === 'R') doAction('reject');
  else if (e.key === 'e' || e.key === 'E') startEdit();
  else if (e.key === 'ArrowRight') go(currentIdx + 1);
  else if (e.key === 'ArrowLeft')  go(currentIdx - 1);
});

// Init
(async () => {
  await loadStats();
  const res = await fetch('/api/next-unreviewed/0');
  const data = await res.json();
  go(data.idx >= 0 ? data.idx : 0);
})();
</script>
</body>
</html>"""

@app.route("/")
def index():
    from flask import Response
    return Response(HTML, mimetype="text/html")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/data/youchengsong/DSL-R1/data/ecommerce/train/queries_product_train_step9.jsonl")
    parser.add_argument("--output", default="/data/youchengsong/DSL-R1/data/ecommerce/train/queries_product_train_review.jsonl")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    INPUT_FILE  = args.input
    OUTPUT_FILE = args.output

    print(f"Loading {INPUT_FILE} ...")
    queries  = load_queries(INPUT_FILE)
    results  = load_results(OUTPUT_FILE)
    print(f"Loaded {len(queries)} queries, {len(results)} already reviewed")
    print(f"Review results → {OUTPUT_FILE}")
    print(f"Open: http://localhost:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
