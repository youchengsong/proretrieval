# LocalSearch-R1 Query 构造流程（最终版 v5）

> 随机分组括号法 + 结构化超集 GT 发现（含全量兜底）

---

## 核心设计

- **复杂度控制：** 叶子数量驱动，N=1~5 对应三级复杂度
- **括号结构：** 随机分组法，覆盖无嵌套 + 所有两层嵌套
- **Gold document：** 组间 OR 每组一个 gold，组间 AND 共享一个 gold
- **GT 发现：** 结构化超集缩小范围，少数情况全量兜底
- **字段选择：** 结构化字段权重更高，自然占多数
- **肯定/否定：** 肯定概率高于否定（Level 1: 7/3, Level 2: 6/4, Level 3: 5/5），无强制约束

---

## 复杂度定义

| N | 采样概率 | Level |
|---|---------|-------|
| 1 | 30% | Level 1（简单）|
| 2 | 20% | Level 2（中等）|
| 3 | 20% | Level 2（中等）|
| 4 | 15% | Level 3（复杂）|
| 5 | 15% | Level 3（复杂）|

---

## 构造流程：九步法

### Step 1：随机叶子数量

随机生成 N ∈ {1, 2, 3, 4, 5}，按上表概率采样。

---

### Step 2：随机分组，确定括号结构和操作符

#### 2a. 随机切割

N-1 个间隙，每个独立地以概率 p 成为切割点，分成 K 组。

| Level | 切割概率 p |
|-------|----------|
| Level 1 (N=1) | — |
| Level 2 (N=2-3) | 30% |
| Level 3 (N=4-5) | 40% |

#### 2b. 掷硬币

| 硬币 | 组内操作符 | 组间操作符 |
|------|----------|----------|
| 正面 (50%) | AND | OR |
| 反面 (50%) | OR | AND |

#### 2c. 特殊情况

- K=1：组内操作符即全局操作符，随机 AND 或 OR
- N=1：无操作符，跳过

---

### Step 3：确定 gold document 数量并采样

| 组间操作符 | Gold 数量 |
|-----------|----------|
| OR | = K（每组一个独立 gold）|
| AND | = 1（所有组共享）|
| 无（K=1）| = 1 |

组间 OR 时不同组的 gold 互不相同，建议同大类内采样。

---

### Step 4：每个组内随机选择字段

从可用字段池中按权重不重复地随机选取。

**电商字段池：**

| 字段 | 模态 | 权重 |
|------|------|------|
| brand | 结构化 | 2 |
| category | 结构化 | 2 |
| price | 结构化 | 2 |
| rating | 结构化 | 2 |
| num_reviews | 结构化 | 1 |
| in_stock | 结构化 | 1 |
| release_date | 结构化 | 1 |
| title | 文本 | 1 |
| description | 文本 | 2 |
| review_summary | 文本 | 2 |
| product_image | 图像 | 2 |

结构化总权重 11，非结构化总权重 7。结构化字段自然占多数。

**组间字段规则：**

| 组间操作符 | 规则 |
|-----------|------|
| OR | 组间字段可重复 |
| AND | 组间字段不重复 |

---

### Step 5：决定肯定 / 否定

每个叶子独立随机，肯定概率高于否定：

| Level | 肯定概率 | 否定概率 |
|-------|---------|---------|
| Level 1 | 70% | 30% |
| Level 2 | 60% | 40% |
| Level 3 | 50% | 50% |

Level 1 以肯定为主，Level 3 肯定与否定各半，否定条件在高复杂度 query 中更常见。无强制约束，极少数情况下某个组内全部否定是允许的，增加数据多样性。

---

### Step 6：填充具体值

每个叶子的值来自其所在组的 gold document。

#### 肯定叶子

| 类型 | 方式 |
|------|------|
| 结构化-类别型 | 等值：`brand = 'Sony'` |
| 结构化-数值型 | 随机比较符 + 阈值（gold 满足）|
| 结构化-布尔型 | 直接用 gold 值 |
| 结构化-日期型 | 范围条件（gold 在范围内）|
| 文本 | LLM 从 gold 文本提取 2-5 个语义词 |
| 图像 | VLM 从 gold 产品图生成视觉描述 |

数值型阈值：`>`/`>=` 取 gold_value × 0.5~0.95，`<`/`<=` 取 gold_value × 1.05~1.5，`=` 概率 10%。

#### 否定叶子

| 类型 | 方式 |
|------|------|
| 结构化 | 取 gold 不具备的值 |
| 文本 | LLM 生成 gold 文本中不存在的特征描述 |
| 图像 | LLM 生成与 gold 外观不同的视觉描述 |

---

### Step 7：编译成 DSL

#### 叶子编译

| 叶子 | 肯定 | 否定 |
|------|------|------|
| 结构化 | `字段 op 值` | `NOT (字段 op 值)` |
| 文本 | `id IN <text_N>` | `id NOT IN <text_N>` |
| 图像 | `id IN <image_N>` | `id NOT IN <image_N>` |

#### 组装

```
K=1:
  叶子1 {组内op} 叶子2 {组内op} ...

K>1 组间 OR:
  (组1叶子 {AND} ...) OR (组2叶子 {AND} ...) OR ...

K>1 组间 AND:
  (组1叶子 {OR} ...) AND (组2叶子 {OR} ...) AND ...

单叶子组不加括号。
```

---

### Step 8：确定完整 GT

构造时的 gold 不是唯一 GT。需执行 DSL 找到全部满足条件的文档。

#### 超集提取规则

| 逻辑结构 | 提取方式 | 是否超集 |
|---------|---------|---------|
| K=1 组内 AND | 删除向量条件，保留结构化 | ✓ 超集 |
| K=1 组内 OR | 无法提取 | 全量执行 |
| K=1 全结构化 | SQL 直接执行 | 精确 GT |
| K>1 组间 OR 组内 AND | 每组删向量保结构化，OR 连接。某组变空则全量 | ✓ 超集（或全量）|
| K>1 组间 AND 组内 OR | 含向量的组整组去掉，保留纯结构化组。全去掉则全量 | ✓ 超集（或全量）|

**为什么外 AND 内 OR 不能直接删向量条件：**

`(A OR <vec>)` 中删掉 `<vec>` 剩 `A`，范围缩小了（砍掉一条通路），不是超集。正确做法是把含向量的整个 OR 组从外层 AND 中去掉（等价于该组变 TRUE），范围变大，保证超集。

#### 执行流程

```
1. 根据逻辑结构提取超集查询
2. 超集有效 → 全量执行超集 SQL → 候选集 → 候选集上执行完整 DSL → GT
3. 超集退化为全量 → 全量执行完整 DSL → GT
```

#### 成本

```
超集有效（~85%）: < 10ms/条
全量执行（~15%）: ~200ms/条（5000 语料库）
6000 条总计: ~4 分钟
```

#### 验证

| 检查项 | 不通过则丢弃重新生成 |
|--------|-------------------|
| GT 非空 | ✓ |
| GT < 语料库 50% | ✓ |

注意：构造时的 gold document 不一定在 GT 中。文本/图像 NOT 条件通过向量检索执行，存在噪音——gold 可能被误匹配到 NOT 的检索结果中而被排除。这是正常现象，不需要丢弃。结构化 NOT 不存在此问题（值是确定性地取 gold 不具备的值）。

---

### Step 9：生成自然语言 + 记录元数据

#### 自然语言

LLM 将 DSL → 口语化查询。OR→"或者"，NOT→"但不要"，AND→逗号。

#### 元数据

```json
{
  "query_id": "q_00001",
  "natural_language_query": "...",
  "gold_dsl": {"sql": "...", "retrieval_list": [...]},
  "gold_document_ids": ["P001", "P002"],
  "gt_document_ids": ["P001", "P002", "P006", "P011"],
  "level": 3,
  "num_leaves": 4,
  "num_groups": 2,
  "group_inner_op": "OR",
  "group_outer_op": "AND",
  "num_not_leaves": 1,
  "modalities_used": ["structured", "text", "image"],
  "logic_pattern": "text AND (struct OR struct OR NOT image)",
  "fields_used": ["review_summary", "brand", "price", "product_image"],
  "gt_discovery_method": "structural_superset",
  "gt_size": 4
}
```

评估使用 `gt_document_ids`。

---

## 走通示例

### 示例 1：Level 3，外 AND 内 OR，单 gold

**Step 1:** N=4, Level 3

**Step 2:**
```
切割: ^1切, ^2不切, ^3不切
分组: [L1] | [L2, L3, L4], K=2
硬币反面 → 组内 OR, 组间 AND
```
骨架: `L1 AND (L2 OR L3 OR L4)`

**Step 3:** 组间 AND → 1 个 gold = P001 (Sony, price=280, rating=4.8, review="excellent ANC, comfortable", 黑色头戴式)

**Step 4:**
```
组1: L1 → review_summary (文本)
组2: L2 → brand (结构化), L3 → price (结构化), L4 → product_image (图像)
```

**Step 5:**
```
L1: 肯定, L2: 肯定, L3: 肯定, L4: 否定
```

**Step 6:**
```
L1: query = "excellent noise cancellation comfortable"
L2: brand = 'Sony'
L3: price < 350
L4(否定): query = "white in-ear earbud with charging case"
```

**Step 7:**
```json
{
  "sql": "SELECT DISTINCT product_id FROM product
          WHERE product_id IN <text_0>
          AND (brand = 'Sony' OR price < 350 OR product_id NOT IN <image_0>)",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "excellent noise cancellation comfortable"},
    {"type": "image", "field": "product_image", "query": "white in-ear earbud with charging case"}
  ]
}
```

**Step 8:**
```
组间 AND, 组内 OR
组1: 只有向量 → 去掉
组2: 含向量 → 去掉
→ 全量执行 → GT = {P001, P006, P019, P042}
```

**Step 9:** "Find products with great noise cancellation reviews, and they should either be Sony, under $350, or at least not look like those white wireless earbuds"

---

### 示例 2：Level 3，外 OR 内 AND，三 gold，超集有效

**Step 1:** N=5, Level 3

**Step 2:**
```
切割: ^2切, ^4切, 其余不切
分组: [L1, L2] | [L3, L4] | [L5], K=3
硬币正面 → 组内 AND, 组间 OR
```
骨架: `(L1 AND L2) OR (L3 AND L4) OR L5`

**Step 3:** 组间 OR → 3 个 gold
```
gold_1 = P001 (Sony, review="excellent ANC")
gold_2 = P002 (Bose, price=320, 银色头戴式)
gold_3 = P005 (Sennheiser, rating=4.9)
```

**Step 4:**
```
组1: L1 → brand (结构化), L2 → review_summary (文本)
组2: L3 → price (结构化), L4 → product_image (图像)
组3: L5 → rating (结构化)
```

**Step 5:** L4 否定，其余肯定

**Step 6:**
```
L1: brand = 'Sony'
L2: query = "excellent noise cancellation"
L3: price < 400
L4(否定): query = "white in-ear earbud with stem"
L5: rating >= 4.5
```

**Step 7:**
```json
{
  "sql": "SELECT DISTINCT product_id FROM product
          WHERE (brand = 'Sony' AND product_id IN <text_0>)
          OR (price < 400 AND product_id NOT IN <image_0>)
          OR rating >= 4.5",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "excellent noise cancellation"},
    {"type": "image", "field": "product_image", "query": "white in-ear earbud with stem"}
  ]
}
```

**Step 8:**
```
组间 OR, 组内 AND
组1: brand='Sony' (删向量)
组2: price<400 (删向量)
组3: rating>=4.5 (纯结构化)
超集: WHERE brand='Sony' OR price<400 OR rating>=4.5
执行 → 候选 ~800
完整 DSL → GT = {P001, P002, P005, P008, P019}
三个 gold 全在 ✓
```

**Step 9:** "I want Sony headphones with great noise cancellation, or anything under $400 that doesn't look like those white earbuds, or any product rated 4.5 and above"

---

### 示例 3：Level 2，外 AND 内 OR，含纯结构化组，超集有效

**Step 1:** N=4, Level 3

**Step 2:**
```
分组: [L1, L2] | [L3, L4], K=2
硬币反面 → 组内 OR, 组间 AND
```

**Step 3:** 组间 AND → 1 gold = P001 (Sony, category=headphone, review="excellent ANC", 黑色头戴式)

**Step 4:**
```
组1: L1 → brand (结构化), L2 → category (结构化)
组2: L3 → review_summary (文本), L4 → product_image (图像)
```

**Step 5:** 全肯定

**Step 6:**
```
L1: brand = 'Sony'
L2: category = 'headphone'
L3: query = "excellent noise cancellation comfortable"
L4: query = "black over-ear headphone cushion pads"
```

**Step 7:**
```json
{
  "sql": "SELECT DISTINCT product_id FROM product
          WHERE (brand = 'Sony' OR category = 'headphone')
          AND (product_id IN <text_0> OR product_id IN <image_0>)",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "excellent noise cancellation comfortable"},
    {"type": "image", "field": "product_image", "query": "black over-ear headphone cushion pads"}
  ]
}
```

**Step 8:**
```
组间 AND, 组内 OR
组1: brand='Sony' OR category='headphone' → 全结构化 → 保留
组2: 含向量 → 去掉
超集: WHERE brand='Sony' OR category='headphone'
执行 → 候选 ~300
完整 DSL → GT = {P001, P006, P012}
```

**Step 9:** "Find Sony products or headphones that have great noise cancellation reviews or look like a black over-ear headphone with cushion pads"

---

### 示例 4：Level 1，单叶子

**Step 1:** N=1
**Step 3:** gold = P001 (rating=4.8)
**Step 4:** L1 → rating (结构化)
**Step 5:** 肯定
**Step 6:** rating >= 4.5

**Step 7:**
```json
{"sql": "SELECT DISTINCT product_id FROM product WHERE rating >= 4.5", "retrieval_list": []}
```

**Step 8:** 纯结构化直接执行 → GT = 所有 rating>=4.5 的产品

**Step 9:** "Find highly rated products, at least 4.5 stars"

---

### 示例 5：Level 2，全 OR，全量执行

**Step 1:** N=2
**Step 2:** K=1, 组内 OR

**Step 3:** 1 gold = P001 (Sony)
**Step 4:** L1 → brand (结构化), L2 → review_summary (文本)
**Step 5:** 全肯定
**Step 6:** L1: brand='Sony', L2: query="excellent noise cancellation"

**Step 7:**
```json
{
  "sql": "SELECT DISTINCT product_id FROM product
          WHERE brand = 'Sony' OR product_id IN <text_0>",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "excellent noise cancellation"}
  ]
}
```

**Step 8:** K=1 组内 OR → 全量执行 → GT = {P001, P002, P005, P006, ...}

**Step 9:** "Find Sony products or anything with great noise cancellation reviews"

---

### 示例 6：Level 2，全 AND 含多模态 + NOT

**Step 1:** N=3
**Step 2:** K=1, 组内 AND
**Step 3:** 1 gold = P002 (Bose, price=320, 银色头戴式, review="best comfort")
**Step 4:** L1→price(结构化), L2→review_summary(文本), L3→product_image(图像)
**Step 5:** L3 否定

**Step 6:**
```
L1: price < 400
L2: query = "comfortable best in class"
L3(否定): query = "black in-ear earbud compact"
```

**Step 7:**
```json
{
  "sql": "SELECT DISTINCT product_id FROM product
          WHERE price < 400 AND product_id IN <text_0> AND product_id NOT IN <image_0>",
  "retrieval_list": [
    {"type": "text", "field": "review_summary", "query": "comfortable best in class"},
    {"type": "image", "field": "product_image", "query": "black in-ear earbud compact"}
  ]
}
```

**Step 8:**
```
K=1 组内 AND → 删向量 → WHERE price<400
超集执行 → 候选 ~2000 → 完整 DSL → GT = {P002, P007, P015}
```

**Step 9:** "Find products under $400 with reviews saying they're the most comfortable, but not the black compact earbud style"

---

## 超集提取规则汇总

| 逻辑结构 | 提取规则 | 结果 |
|---------|---------|------|
| K=1 组内 AND | 删向量条件，保留结构化 | 超集 |
| K=1 组内 OR | 无法提取 | 全量 |
| K=1 全结构化 | 直接执行 | 精确 GT |
| K>1 组间 OR 组内 AND | 每组删向量保结构化，OR 连接。某组变空则全量 | 超集或全量 |
| K>1 组间 AND 组内 OR | 去掉含向量的组，保留纯结构化组。全去掉则全量 | 超集或全量 |

---

## Evaluation

### 候选池

每条 query 固定 500 个文档，所有方法同一候选池：

```
GT 文档 + 结构化重叠 negatives + 文本近邻 negatives + 图像近邻 negatives + 随机补足
```

### 指标

主表: Hit@1, Hit@3, MRR
分析: Recall@k, nDCG@5

---

## 数据集统计表

| 指标 | 电商 | 邮件 |
|------|------|------|
| 语料库 | ~5,000 | ~2,500 |
| Train / Eval | 4,000 / 1,000 | 2,000 / 500 |
| Level 1 / 2 / 3 | 30% / 40% / 30% | 30% / 40% / 30% |
| 外OR内AND | TBD | TBD |
| 外AND内OR | TBD | TBD |
| 全AND | TBD | TBD |
| 全OR | TBD | TBD |
| 含 NOT | TBD | TBD |
| 含文本语义 | TBD | TBD |
| 含图像语义 | TBD | — |
| 单 gold | TBD | TBD |
| 多 gold | TBD | TBD |
| 平均 GT 大小 | TBD | TBD |
| 超集有效比例 | TBD | TBD |
| 全量执行比例 | TBD | TBD |
