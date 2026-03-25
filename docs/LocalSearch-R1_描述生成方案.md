# LocalSearch-R1 文本/图像描述生成方案

> 为 query 构造 Step 6 中的文本（text）和图像（image_url）叶子预生成 lookup table

---

## 背景

Query 构造 Step 6 中，文本和图像叶子的具体值需要调用大模型生成：

| 叶子类型 | 肯定值 | 否定值 |
|---------|--------|--------|
| text（肯定）| 从 gold 的 text 字段提取 3-5 个语义关键词/短语 | — |
| text（否定）| — | 同 category 内语义差异足够大的其他商品描述 |
| image_url（肯定）| 描述 gold 商品图像的外观 | — |
| image_url（否定）| — | 描述与 gold 图像外观不同的商品 |

**策略：离线预生成 lookup table**，query 构造时直接查表，不实时调用 API。

---

## 四条描述的生成方法

### 1. `text_pos`（文本肯定）

**模型**：`deepseek-v3-2-251201`

**输入**：商品 `text` 字段（title + bullet_point 拼接）

**Prompt**：
```
Product description:
{text}

Extract 3-5 key semantic phrases (as a short search query, space-separated) that describe this product's core features, use case, and materials.
Output only the query, no explanation.
```

**示例输出**：`"braided waxed rodeo lasso leather burner kids"`

---

### 2. `text_neg`（文本否定）

**方法**：跨商品采样 + 相似度过滤，fallback 调 API

**流程**：
1. 从同 category 中随机抽取另一个商品的 `text_pos` 作为候选
2. 用 Qwen3-Embedding-8B 计算候选与 gold `text_pos` 的余弦相似度
3. 相似度 **< 0.4** → 接受，否则继续采样（最多 30 个候选）
4. 30 个候选均不满足 → fallback 到 API 生成

**Fallback Prompt（`deepseek-v3-2-251201`）**：
```
Product category: {category}
Product features: {text_pos}

Describe a DIFFERENT product in the same category (5-8 words) with clearly different features, materials, or use case.
Output ONLY the description, no explanation.
```

---

### 3. `image_pos`（图像肯定）

**模型**：`gemini-2.5-flash`（支持视觉）

**流程**：
1. 图片已预先下载到本地：`/data/youchengsong/DSL-R1/image/{product_id}.jpg`
2. base64 编码后传给 Gemini API
3. 生成简短的视觉描述

**Prompt**：
```
Describe this product image in 5–10 words as a concise e-commerce search query.
Focus on the product type, key visual features, color, and material.
Output ONLY the query, no explanation.
```

**示例输出**：`"pink handheld metal punch tool with grip handle"`

---

### 4. `image_neg`（图像否定）

**方法**：跨商品采样 + 相似度过滤，fallback 调 API

**流程**：
1. 从同 category 中随机抽取另一个商品的 `image_pos` 作为候选
2. 用 Qwen3-Embedding-8B 计算候选与 gold `image_pos` 的余弦相似度
3. 相似度 **< 0.4** → 接受，否则继续采样（最多 30 个候选）
4. 30 个候选均不满足 → fallback 到 API 生成

**Fallback Prompt（`gemini-2.5-flash`）**：
```
Category: {category}
Reference product: {image_pos}

Describe a DIFFERENT product in the same category (5–8 words)
with clearly different appearance, color, or style.
Output ONLY the description, no explanation.
```

---

## 执行顺序

```
Phase A: 生成 text_pos（API）
  → deepseek-v3-2-251201，13,334 次调用
  → 输出：text_pos.jsonl

Phase B: 生成 text_neg（本地 + fallback）
  → Qwen3-Embedding-8B（GPU:2）嵌入所有 text_pos
  → 同 category 采样 + 余弦相似度过滤（阈值 0.4）
  → fallback：deepseek-v3-2-251201（预计 ~20%）
  → 输出：text_descriptions.jsonl

Phase C: 下载图片（一次性）
  → 下载到 /data/youchengsong/DSL-R1/image/{product_id}.jpg
  → 脚本：download_images.py

Phase D: 生成 image_pos（API + 视觉）
  → gemini-2.5-flash，读本地图片 base64 编码后调用
  → 输出：image_pos.jsonl

Phase E: 生成 image_neg（本地 + fallback）
  → Qwen3-Embedding-8B（GPU:2）嵌入所有 image_pos
  → 同 category 采样 + 余弦相似度过滤（阈值 0.4）
  → fallback：gemini-2.5-flash（预计 ~20%）
  → 输出：image_descriptions.jsonl
```

---

## 工程设计

| 设计点 | 方案 |
|--------|------|
| **断点续传** | 实时 append 写 JSONL，已生成的 product_id 自动跳过 |
| **并发** | asyncio，文本 concurrency=20，图像 concurrency=10 |
| **图片存储** | 预先下载到本地磁盘，生成描述时从磁盘读取 |
| **失败处理** | 3 次重试，最终失败记录到日志 |
| **相似度计算** | Qwen3-Embedding-8B，GPU 固定为 2 号卡 |

---

## 成本估算（13,334 个商品）

| 步骤 | 模型 | 调用次数 | 小计 |
|------|------|---------|------|
| text_pos | deepseek-v3-2-251201 | 13,334 | ~$0.98 |
| text_neg fallback | deepseek-v3-2-251201 | ~2,667 | ~$0.19 |
| image_pos | gemini-2.5-flash | 13,334 | ~$0.65 |
| image_neg fallback | gemini-2.5-flash | ~2,667 | ~$0.11 |
| text_neg / image_neg sampling | Qwen3-Embedding-8B（本地）| — | $0 |
| **合计** | | **~32,000 次** | **≈ $1.93** |

---

## 输出文件

```
/data/youchengsong/DSL-R1/data/
├── text_pos.jsonl            # 中间文件：{"product_id", "text_pos"}
├── text_pos_emb.npy          # 中间文件：text_pos 向量矩阵
├── text_pos_emb_ids.json     # 中间文件：向量行顺序对应的 product_id 列表
├── text_descriptions.jsonl   # 最终输出：{"product_id", "text_pos", "text_neg"}
├── image_pos.jsonl           # 中间文件：{"product_id", "image_pos"}
├── image_pos_emb.npy         # 中间文件：image_pos 向量矩阵
├── image_pos_emb_ids.json    # 中间文件：向量行顺序对应的 product_id 列表
└── image_descriptions.jsonl  # 最终输出：{"product_id", "image_pos", "image_neg"}
```

**最终格式示例**：
```jsonl
// text_descriptions.jsonl
{"product_id": "B07PLF39YR", "text_pos": "handheld metal hole punch single circle pink grip", "text_neg": "heavy duty stapler with magazine loading"}

// image_descriptions.jsonl
{"product_id": "B07PLF39YR", "image_pos": "pink handheld metal punch tool with grip handle", "image_neg": "black electric desktop stapler with tray"}
```

---

# 邮件数据集描述生成方案（Enron Email）

> 为 Enron 邮件数据集（enron_final.parquet，10,000 条）预生成 lookup table

## 四条描述的生成方法

邮件有两个文本字段：`Subject`（主题）和 `Message`（正文），各自生成肯定和否定描述。

| 描述 | 输入 | 说明 |
|------|------|------|
| `subject_pos` | Subject 原文 | 能检索到该邮件主题的搜索 query（3-8 词） |
| `subject_neg` | subject_pos | 完全不同主题的搜索 query（3-8 词） |
| `message_pos` | Message 正文（截断到 1000 字符） | 能检索到该邮件内容的搜索 query（8-15 词） |
| `message_neg` | message_pos | 完全不同内容的搜索 query（8-15 词） |

**模型**：`deepseek-v3-2-251201`（全部四条均使用文本 API）

### Prompts

**subject_pos**：
```
Email subject: "{subject}"

Write a short search query (3-8 words) that a user might type to find an email about this topic.
Output ONLY the query, no explanation, no punctuation.
```

**message_pos**（Message 截断到前 1000 字符）：
```
Email content:
{message}

Write a short search query (8-15 words) that captures the main topic and key information of this email.
Output ONLY the query, no explanation, no punctuation.
```

**subject_neg**：
```
Subject search query: "{subject_pos}"

Write a search query (3-8 words) for an email about a COMPLETELY DIFFERENT topic.

Examples:
- pos: "quarterly budget review finance" → neg: "employee onboarding training schedule"
- pos: "power plant maintenance outage" → neg: "client dinner reservation confirmation"
- pos: "contract negotiation legal terms" → neg: "vacation leave request approval"

Output ONLY the query, no explanation, no punctuation.
```

**message_neg**：
```
Message search query: "{message_pos}"

Write a search query (8-15 words) for an email with COMPLETELY DIFFERENT content and topic.

Examples:
- pos: "natural gas pipeline capacity trading deal western region" → neg: "employee performance review feedback mid-year evaluation form"
- pos: "meeting agenda board directors quarterly earnings report" → neg: "software system upgrade deployment schedule IT infrastructure"
- pos: "client invoice payment overdue reminder accounting department" → neg: "team building event outdoor activity planning logistics"

Output ONLY the query, no explanation, no punctuation.
```

## 执行流程

```
Phase 1: 并发生成 subject_pos + message_pos
  → deepseek-v3-2-251201，20,000 次调用（10,000 × 2）
  → 输出：subject_pos.jsonl + message_pos.jsonl

Phase 2: 并发生成 subject_neg + message_neg
  → deepseek-v3-2-251201，20,000 次调用（10,000 × 2）
  → 输出：email_descriptions.jsonl
```

## 工程设计

| 设计点 | 方案 |
|--------|------|
| **断点续传** | Phase 1 实时 append 写中间 JSONL，Phase 2 整体重写输出文件 |
| **并发** | asyncio，concurrency=20 |
| **Message 截断** | 只取前 1000 字符，避免超长邮件浪费 token |
| **失败处理** | 3 次重试，指数退避，最终失败记为 null |
| **无嵌入过滤** | neg 直接由 API 生成，不使用 embedding 相似度过滤 |

## 输出文件

```
/data/youchengsong/DSL-R1/data/
├── subject_pos.jsonl          # 中间文件：{"message_id", "subject_pos"}
├── message_pos.jsonl          # 中间文件：{"message_id", "message_pos"}
└── email_descriptions.jsonl   # 最终输出：{"message_id", "subject_pos", "subject_neg", "message_pos", "message_neg"}
```

**最终格式示例**：
```jsonl
{
  "message_id": "<abc123@enron.com>",
  "subject_pos": "natural gas trading contract western pipeline",
  "subject_neg": "employee vacation request approval HR",
  "message_pos": "Enron energy trading deal natural gas pipeline capacity west coast",
  "message_neg": "employee performance review mid-year feedback form submission"
}
```

**脚本**：`/data/youchengsong/DSL-R1/src/email/gen_email_descriptions.py`
