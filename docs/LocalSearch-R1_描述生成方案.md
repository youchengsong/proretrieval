# LocalSearch-R1 文本/图像描述生成方案

> 为 query 构造 Step 6 中的文本（text）和图像（image_url）叶子预生成 lookup table

---

## 背景

Query 构造 Step 6 中，文本和图像叶子的具体值需要调用大模型生成：

| 叶子类型 | 肯定值 | 否定值 |
|---------|--------|--------|
| text（肯定）| 从 gold 的 text 字段提取 2-5 个语义关键词/短语 | — |
| text（否定）| — | 描述 gold 没有的特征 |
| image_url（肯定）| 描述 gold 商品图像的外观 | — |
| image_url（否定）| — | 描述与 gold 图像外观不同的商品 |

**策略：离线预生成 lookup table**，query 构造时直接查表，不实时调用 API。

---

## 四条描述的生成方法

### 1. `text_pos`（文本肯定）

**模型**：DeepSeek-V3

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

**方法**：跨商品采样 + 相似度过滤（零 API 调用）

**流程**：
1. 从同 category 中随机抽取另一个商品的 `text_pos` 作为候选 `text_neg`
2. 用 sentence-transformers 计算候选与 gold `text_pos` 的余弦相似度
3. 相似度 < 0.4 → 接受；0.4~0.6 → 标记质量低；> 0.6 → 拒绝重采样
4. 同 category 内找不到合适候选 → fallback 到 LLM 生成

**Fallback Prompt（DeepSeek-V3）**：
```
Product category: {category}
Product description: {text}

Generate a short description (5-8 words) of a DIFFERENT product in the same category
that has clearly different features or use case.
Output only the description.
```

---

### 3. `image_pos`（图像肯定）

**模型**：Gemini 1.5 Flash（支持视觉）

**流程**：下载 `image_url` 图片 → 传给 Gemini → 生成视觉描述

**Prompt**：
```
Describe this product's appearance in 6-10 words, focusing on:
color, shape, material, and key visual characteristics.
Output only the description, no explanation.
```

**示例输出**：`"pink handheld metal punch tool with grip handle"`

---

### 4. `image_neg`（图像否定）

**方法**：跨商品采样 + 相似度过滤（零 API 调用）

**流程**：
1. 从同 category 中随机抽取另一个商品的 `image_pos` 作为候选 `image_neg`
2. 计算候选与 gold `image_pos` 字符串的余弦相似度（sentence-transformers）
3. 相似度 < 0.4 → 接受；> 0.6 → 拒绝重采样
4. Fallback → DeepSeek-V3 基于 `image_pos` + `category` 生成对立描述

**Fallback Prompt（DeepSeek-V3）**：
```
A product in category "{category}" looks like: "{image_pos}"

Describe what a DIFFERENT product in the same category looks like (6-10 words),
with clearly different color, shape, or form factor.
Output only the description.
```

---

## 执行顺序

```
Phase 0（一次性离线预生成）

Step A: 生成所有 text_pos
  → DeepSeek-V3，13,334 次调用

Step B: 生成所有 image_pos
  → 下载图片 + Gemini 1.5 Flash，13,334 次调用

Step C: 批量匹配 text_neg 和 image_neg
  → sentence-transformers 本地计算相似度，0 次 API
  → fallback 部分调用 DeepSeek-V3（预计 ~10-20%）

输出：product_descriptions.jsonl
  每行：{"product_id": "...", "text_pos": "...", "text_neg": "...", "image_pos": "...", "image_neg": "..."}
```

---

## 工程设计

| 设计点 | 方案 |
|--------|------|
| **断点续传** | 实时 append 写 JSONL，已生成的 product_id 跳过 |
| **并发** | asyncio 并发 20 个请求，控制速率避免限流 |
| **图片处理** | 下载到内存（不落盘），base64 编码后传给 Gemini |
| **失败处理** | 3 次重试，超时 30s，最终失败记录到 failed_ids.txt |
| **相似度计算** | 使用 `sentence-transformers/all-MiniLM-L6-v2`，本地运行 |

---

## 成本估算（13,334 个商品）

### Token 估算

| 步骤 | 输入 tokens/次 | 输出 tokens/次 |
|------|--------------|--------------|
| text_pos | ~210（product text + prompt）| ~15 |
| image_pos | ~566（图片 ~516 + prompt ~50）| ~20 |
| text_neg fallback（~20%）| ~200 | ~15 |
| image_neg fallback（~20%）| ~100 | ~15 |

### 费用明细

| 步骤 | 模型 | 调用次数 | 输入费用 | 输出费用 | 小计 |
|------|------|---------|---------|---------|------|
| text_pos | DeepSeek-V3 | 13,334 | $0.27/1M × 2.8M = **$0.76** | $1.10/1M × 0.2M = **$0.22** | **$0.98** |
| text_neg sampling | — | 0 | $0 | $0 | **$0** |
| text_neg fallback | DeepSeek-V3 | ~2,667 | **$0.15** | **$0.04** | **$0.19** |
| image_pos | Gemini 1.5 Flash | 13,334 | $0.075/1M × 7.5M = **$0.57** | $0.30/1M × 0.27M = **$0.08** | **$0.65** |
| image_neg sampling | — | 0 | $0 | $0 | **$0** |
| image_neg fallback | DeepSeek-V3 | ~2,667 | **$0.07** | **$0.04** | **$0.11** |
| **合计** | | **~32,000次** | | | **≈ $1.93** |

> 仅 corpus 5,000 商品的成本约 $0.70；全量 13,334 约 $1.93。

---

## 输出格式

```jsonl
{"product_id": "B07PLF39YR", "text_pos": "handheld metal hole punch single circle pink grip", "text_neg": "heavy duty stapler with magazine loading", "image_pos": "pink handheld metal punch tool with grip handle", "image_neg": "black electric desktop stapler with tray"}
{"product_id": "B00118YOYI", "text_pos": "ammonia free hair lightener vegan balayage highlights", "text_neg": "black permanent hair dye dark coverage cream", "image_pos": "white box hair dye kit with developer bottle", "image_neg": "dark brown tube hair color cream packaging"}
```
