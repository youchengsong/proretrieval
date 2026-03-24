import os
import json
import random
import sqlite3
from datetime import date, datetime
import asyncio
from typing import List, Dict, Union
import re

# 设置Google Cloud认证
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/szSFT/thinredline-gemini-5fa8c502b645.json"

# 使用 google-genai 库
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

# 移除不需要的 langchain 导入，创建兼容的简单类
class SimplePromptTemplate:
    def __init__(self, template):
        self.template = template
    
    @classmethod
    def from_template(cls, template):
        return cls(template)
    
    def __or__(self, other):
        return SimpleChain(self, other)

class SimpleOutputParser:
    pass

class SimpleChain:
    def __init__(self, prompt, model):
        self.prompt = prompt
        self.model = model
    
    def __or__(self, parser):
        return SimpleChain(self.prompt, self.model)
    
    def invoke(self, params):
        # 格式化模板
        formatted_prompt = self.prompt.template.format(**params)
        # 调用模型
        response = self.model(formatted_prompt)
        return response

# 创建兼容的类实例
ChatPromptTemplate = SimplePromptTemplate
StrOutputParser = SimpleOutputParser

# Vertex AI 配置
PROJECT_ID = "thinredline-gemini"
LOCATION = "us-central1"

# 初始化Vertex AI客户端
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

def genai(temperature=0.5, top_p=0.4):
    """返回配置好的模型生成函数"""
    def generate_content(prompt_text):
        contents = [
            {
                "role": "user", 
                "parts": [{"text": prompt_text}]
            }
        ]
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=GenerateContentConfig(
                temperature=temperature,
                thinking_config=ThinkingConfig(
                    thinking_budget=150,
                ),
            )
        )
        return response.text
    
    return generate_content

# 数据库路径
DB_PATH = "/workspace/zyr_copy/mail-ai-lib/db/emails_2300.db"

# 主线程连接数据库（用于非并行操作）
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 关系模板：仅针对结构化字段
RELATION_TEMPLATES = [
    "(在账户 <account_email> 下)",
    "(收件于 <received_date>)",
    "(草稿邮件，创建于 <draft_created_date>)",
    "(草稿邮件，修改于 <draft_modified_date>)",
    "(已读邮件)",
    "(未读邮件)",
    "(已归档邮件)",
    "(未归档邮件)",
    "(已转发邮件)",
    "(未转发邮件)",
    "(已回复邮件)",
    "(未回复邮件)",
    "(星标邮件)",
    "(非星标邮件)",
    "(重要紧急邮件)",
    "(非重要紧急邮件)",
    "(所在对话共 <thread_message_count> 封邮件)",
]


def sample_relational_requirements():
    """随机抽取一条邮件，仅针对结构化字段生成关系需求"""
    cursor.execute("""
        SELECT account_email, received_date,
               is_draft, draft_created_date, draft_modified_date,
               is_read, is_starred, is_archived, is_forwarded, is_replied, is_prioritized,
               thread_message_count
        FROM email
        ORDER BY RANDOM()
        LIMIT 1
    """)
    record = cursor.fetchone()
    (account_email, received_date,
     is_draft, draft_created_date, draft_modified_date,
     is_read, is_starred, is_archived, is_forwarded, is_replied, is_prioritized,
     thread_message_count) = record

    # 生成候选模板
    candidates = []
    if account_email:
        candidates.append("(在账户 <account_email> 下)")
    if received_date:
        try:
            # 标准化 ISO 8601 时间字符串
            if received_date.endswith("Z"):
                received_date_format = received_date[:-1] + "+00:00"
            else:
                received_date_format = received_date
            dt = datetime.fromisoformat(received_date_format)
            rd_date = dt.date()
            days_diff = (date.today() - rd_date).days

            # 引入偏移随机天数（-2 到 +3）
            offset = random.randint(0, 3)
            approx_days = max(0, days_diff + offset)

            # 随机决定展示格式（相对 / 绝对）
            use_relative = random.choice([True, False])

            if use_relative:
                candidates.append(f"(收件于 最近{approx_days}天，今天日期：{date.today()})")
            else:
                candidates.append(f"(收件于 {received_date}，今天日期：{date.today()})")

        except Exception as e:
            candidates.append(f"(收件于 {received_date}，今天日期：{date.today()})")
    
    if is_draft:
        if draft_created_date:
            candidates.append("(草稿邮件，创建于 <draft_created_date>)")
        if draft_modified_date:
            candidates.append("(草稿邮件，修改于 <draft_modified_date>)")
    if is_read == 1:
        candidates.append("(已读邮件)")
    elif is_read == 0:
        candidates.append("(未读邮件)")
    if is_archived == 1:
        candidates.append("(已归档邮件)")
    elif is_archived == 0:
        candidates.append("(未归档邮件)")
    if is_forwarded == 1:
        candidates.append("(已转发邮件)")
    elif is_forwarded == 0:
        candidates.append("(未转发邮件)")
    if is_replied == 1:
        candidates.append("(已回复邮件)")
    elif is_replied == 0:
        candidates.append("(未回复邮件)")
    if is_starred == 1:
        candidates.append("(星标邮件)")
    elif is_starred == 0:
        candidates.append("(非星标邮件)")
    if is_prioritized == 1:
        candidates.append("(重要紧急邮件)")
    elif is_prioritized == 0:
        candidates.append("(非重要紧急邮件)")
    if thread_message_count is not None:
        candidates.append("(所在对话共 <thread_message_count> 封邮件)")

    # 回退到默认模板
    if not candidates:
        candidates = RELATION_TEMPLATES.copy()
        print("[WARN] 无可用结构化模板，使用默认列表")

    # 随机选取 1～2 条
    k = random.randint(1, min(2, len(candidates)))
    selected = random.sample(candidates, k)

    # 占位符替换
    # print(f'[STEP1] selected templates: {selected}')
    rel_reqs = []
    for tpl in selected:
        text = tpl.replace("<account_email>", account_email or "")
        text = text.replace("<received_date>", received_date or "")
        text = text.replace("<draft_created_date>", draft_created_date or "")
        text = text.replace("<draft_modified_date>", draft_modified_date or "")
        text = text.replace("<thread_message_count>", str(thread_message_count or ""))
        rel_reqs.append(text)
    # print(f"[STEP1] 结构化关系需求: {rel_reqs}")
    return rel_reqs

def rel_reqs_to_sql(rel_reqs):
    """将关系需求列表转换为 SQL WHERE 条件串联 仅针对结构化字段。"""
    conds = []
    for req in rel_reqs:
        if m := re.search(r"在账户\s*([^ )]+)", req):
            conds.append(f"account_email = '{m.group(1)}'")
        elif m := re.search(r"收件于\s*([^ )]+)，", req):
            part = m.group(1)
            if "最近" in part:
                days = int(re.search(r"最近(\d+)天", part).group(1) or 7)
                conds.append(f"received_date >= date('now','-{days} day')")
            else:
                conds.append(f"received_date = '{part}'")
        elif "草稿邮件" in req:
            conds.append("is_draft = 1")
            if m2 := re.search(r"创建于\s*([^ )]+)", req):
                conds.append(f"draft_created_date = '{m2.group(1)}'")
            if m2 := re.search(r"修改于\s*([^ )]+)", req):
                conds.append(f"draft_modified_date = '{m2.group(1)}'")
        elif "已读邮件" in req:
            conds.append("is_read = 1")
        elif "未读邮件" in req:
            conds.append("is_read = 0")
        elif "已归档邮件" in req:
            conds.append("is_archived = 1")
        elif "未归档邮件" in req:
            conds.append("is_archived = 0")
        elif "已转发邮件" in req:
            conds.append("is_forwarded = 1")
        elif "未转发邮件" in req:
            conds.append("is_forwarded = 0")
        elif "已回复邮件" in req:
            conds.append("is_replied = 1")
        elif "未回复邮件" in req:
            conds.append("is_replied = 0")
        elif "星标邮件" in req:
            conds.append("is_starred = 1")
        elif "非星标邮件" in req:
            conds.append("is_starred = 0")
        elif "重要紧急邮件" in req:
            conds.append("is_prioritized = 1")
        elif "非重要紧急邮件" in req:
            conds.append("is_prioritized = 0")
        elif m := re.search(r"所在对话共\s*(\d+)", req):
            conds.append(f"thread_message_count = {m.group(1)}")
    return " AND ".join(conds)

# 文本属性抽取：针对非结构化字段（线程安全版本）
def extract_textual_requirements(email_id):
    # 为每个线程创建独立的数据库连接
    thread_conn = sqlite3.connect(DB_PATH)
    thread_cursor = thread_conn.cursor()
    
    try:
        thread_cursor.execute("""
            SELECT sender_email, sender_name,
                   recipient_list, cc_list, bcc_list,
                   subject, email_content,
                   folder_labels, attachment_list
            FROM email WHERE message_id = ?
        """, (email_id,))
        row = thread_cursor.fetchone()
        if not row:
            raise ValueError(f"No email found with id {email_id!r}")
        sender_email, sender_name, recipient_list, cc_list, bcc_list, subject, content, folder_labels, attachment_list = row

        info = (
            f"发件人邮箱: {sender_email}\n"
            f"发件人姓名: {sender_name}\n"
            f"收件人列表: {recipient_list}\n"
            f"抄送列表: {cc_list}\n"
            f"密送列表: {bcc_list}\n"
            f"文件夹标签: {folder_labels}\n"
            f"附件列表: {attachment_list}\n"
            f"主题: {subject}\n正文: {content}"
        )
        print(f"[STEP2] 非结构化信息前200字:\n{info[:200]!r}…")

        extract_count = random.randint(1, 3)
        template = """下面是一封邮件的非结构化信息（发件人/收件人/主题/正文/文件夹/附件）：
{info}

请随机抽取 {count} 条"文本检索"需求（示例："包含项目预算讨论"、"含有会议邀请附件"），
只需输出 JSON 列表，例如：
[
  {{ "附件列表": "Sharing images from recent trip" }}
]
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | genai() | StrOutputParser()
        result = chain.invoke({"info": info, "count": extract_count})
        print('*'*50)
        print(f"[STEP2] 抽取 {extract_count} 条文本需求:\n{result}")
        print(result)
        return result
    finally:
        thread_cursor.close()
        thread_conn.close()

def compose_queries(rel_reqs, props):
    """
    STEP 3：合成查询
    - rel_reqs: 关系需求列表，如 ["账户名: foo", "发件人: bar"]
    - props: 文本属性字符串
    返回：(初步查询, 丰富后查询)
    """
    # 1. 拼接关系需求
    rel_text = "，且".join(rel_reqs)

    # 2. 初步生成
    template1 = """关系需求：{rel_text}
文本属性：{props}

请生成一条严谨、清晰和专业的自然语言检索查询，要求：
- 如果包含“账户名”、“发件人”、“收件人”等信息，生成的检索语句不要描述错误；
- 检索内容可以对文本属性使用同义或近义表述，以增强检索覆盖；
- 不应该过度改变原意，但可以适度调整文本属性措辞；
- 关系需求与文本属性应自然融合；
- 需求的所属类别应明确，且不可改变原意；
- 查询应涵盖上述所有需求，清晰明确的自然语言检索查询。
"""
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain1 = prompt1 | genai() | StrOutputParser()
    r1 = chain1.invoke({"rel_text": rel_text, "props": props})
    # print(f"[STEP3] 初步查询: {r1}")

    # 3. 风格丰富与润色
    template2 = """你是一位资深邮件检索专家，擅长模拟多种用户风格的查询表达。

原始信息：
关系需求：{rel_text}
文本属性：{props}

初步查询："{initial_query}"

请在保持核心意图的前提下：
- 如果包含“账户名”、“发件人”、“收件人”等信息，生成的检索语句不要描述错误；
- 丰富上下文信息，使语言更贴近真实用户的语气和风格；
- 避免过度精准的时间描述，适度模糊时效；
- 控制查询长度，保持语言自然流畅；
- 输出应为中文或英文之一；
- 不生成与原始信息无关的内容；
- 文本属性的表达方式应当有一点变化，可以增加或者减少一些细节，让语言风格更像真实用户；
- 在不改变其本意的前提下，调整文本属性的表达方式，增加检索的难度；
- 关系需求中的字段顺序可调整，表达方式可灵活改写（如“发件人邮箱: xxx”可写为“来自 xxx 的邮件”），但字段中的具体值不得修改；
- 仅返回检索语句，不要附加其他说明。
"""
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = prompt2 | genai() | StrOutputParser()
    r2 = chain2.invoke({
        "rel_text": rel_text,
        "props": props,
        "initial_query": r1
    })

    return r1, r2

def filter_candidates(rel_reqs, text_reqs, candidates):
    """
    对候选列表执行双重验证：
    1. SQL 结构化过滤已在 pipeline 中完成。
    2. 使用 LLM 验证非结构化文本需求。
    返回满足所有需求的 message_id 列表。
    注意：本函数在并行线程中调用，需使用线程内独立的 SQLite 连接。
    """
    # 线程内独立连接，避免跨线程使用全局 cursor
    thread_conn = sqlite3.connect(DB_PATH)
    thread_cursor = thread_conn.cursor()

    try:
        final = []

        # 定义 PromptTemplate
        template = """邮件 ID：{msg_id}

-- 结构化信息 --
账户邮箱: {account_email}
收件时间: {received_date}
草稿: {is_draft} (创建于{draft_created_date}，修改于{draft_modified_date})
已读: {is_read}，星标: {is_starred}，归档: {is_archived}，转发: {is_forwarded}，回复: {is_replied}，重要性: {is_prioritized}
对话长度: {thread_message_count}

-- 关系需求 --
{rel_text}

-- 非结构化信息 --
发件人邮箱: {sender_email}
发件人姓名: {sender_name}
收件人列表: {recipient_list}
抄送列表: {cc_list}
密送列表: {bcc_list}
文件夹标签: {folder_labels}
附件列表: {attachment_list}
主题: {subject}
正文: {content}

-- 文本需求 --
{text_reqs}

请判断此邮件是否同时满足上述“关系需求”与“文本需求”，回答“是”或“否”，并简要说明理由。"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | genai() | StrOutputParser()

        for msg_id in candidates:
            # 查询邮件所有字段（使用线程内 cursor）
            thread_cursor.execute(
                """
                SELECT account_email, received_date,
                       is_draft, draft_created_date, draft_modified_date,
                       is_read, is_starred, is_archived, is_forwarded, is_replied, is_prioritized,
                       thread_message_count,
                       sender_email, sender_name,
                       recipient_list, cc_list, bcc_list,
                       subject, email_content,
                       folder_labels, attachment_list
                FROM email
                WHERE message_id = ?
                """,
                (msg_id,)
            )
            row = thread_cursor.fetchone()
            if not row:
                continue

            (
                account_email, received_date,
                is_draft, draft_created_date, draft_modified_date,
                is_read, is_starred, is_archived, is_forwarded, is_replied, is_prioritized,
                thread_message_count,
                sender_email, sender_name,
                recipient_list, cc_list, bcc_list,
                subject, content,
                folder_labels, attachment_list
            ) = row

            rel_text = "；".join(rel_reqs)

            # 调用 LLM 验证
            response = chain.invoke({
                "msg_id": msg_id,
                "account_email": account_email,
                "received_date": received_date,
                "is_draft": is_draft,
                "draft_created_date": draft_created_date,
                "draft_modified_date": draft_modified_date,
                "is_read": is_read,
                "is_starred": is_starred,
                "is_archived": is_archived,
                "is_forwarded": is_forwarded,
                "is_replied": is_replied,
                "is_prioritized": is_prioritized,
                "thread_message_count": thread_message_count,
                "sender_email": sender_email,
                "sender_name": sender_name,
                "recipient_list": recipient_list,
                "cc_list": cc_list,
                "bcc_list": bcc_list,
                "folder_labels": folder_labels,
                "attachment_list": attachment_list,
                "subject": subject,
                "content": content,
                "rel_text": rel_text,
                "text_reqs": text_reqs
            }).strip()

            verdict = response.splitlines()[0]
            print(f"{msg_id} 验证结果: {verdict}")
            if verdict.startswith("是"):
                final.append(msg_id)

        print(f"[STEP4] 通过验证的候选: {final}")
        return final
    finally:
        try:
            thread_cursor.close()
        finally:
            thread_conn.close()

def parse_text_requirements(text_requirements: str) -> List[Dict[str, str]]:
    """
    解析 text_requirements 中的 JSON 格式数据
    
    Args:
        text_requirements (str): 包含JSON的字符串，形如 "```json\n[...]\n```"
        
    Returns:
        List[Dict[str, str]]: 解析后的需求列表
    """
    import json
    
    try:
        # 移除 ```json 和 ``` 标记
        cleaned = text_requirements.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # 移除 ```json
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]  # 移除 ```
        
        cleaned = cleaned.strip()
        
        # 解析JSON
        requirements = json.loads(cleaned)
        return requirements if isinstance(requirements, list) else [requirements]
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"解析 text_requirements JSON 失败: {e}")
        print(f"原始内容: {text_requirements}")
        return []


def text_requirements_to_sql_vector(text_requirements: str) -> Dict[str, Union[str, List[Dict]]]:
    """
    将text_requirements转换为SQL查询和向量查询列表
    text_requirements只包含非结构化信息，全部转为向量查询
    
    Args:
        text_requirements (str): JSON格式的文本需求，形如 "```json\n[{\"主题\": \"会议\"}]\n```"
        
    Returns:
        Dict: 包含 'sql' 和 'vector_query_list' 的字典
              - sql: SQL查询语句，包含<vector_N>占位符
              - vector_query_list: 向量查询列表，每个元素是字段名到查询值的映射
    """
    import json
    
    # 解析text_requirements中的JSON
    requirements = parse_text_requirements(text_requirements)
    
    if not requirements:
        # 如果解析失败，将整个字符串作为邮件内容查询
        return {
            "sql": "SELECT message_id FROM email WHERE message_id IN <vector_0>",
            "vector_query_list": [{"email_content": text_requirements}]
        }
    
    # 中文字段名到数据库字段名的映射表
    field_mapping = {
        "主题": "subject",
        "发件人邮箱": "sender_email",
        "发件人姓名": "sender_name",
        "收件人列表": "recipient_list",
        "抄送列表": "cc_list",
        "密送列表": "bcc_list", 
        "正文": "content",
        "文件夹标签": "folder_labels",
        "附件列表": "attachment_list",
    }
    
    try:
        vector_queries = []
        
        # 处理每个需求项
        for req in requirements:
            for chinese_field, value in req.items():
                # 将中文字段名映射到英文数据库字段名
                english_field = field_mapping.get(chinese_field, chinese_field.lower())
                
                # 添加到向量查询列表
                vector_queries.append({english_field: value})
        
        # 构建SQL查询 - 所有条件都使用向量查询
        if vector_queries:
            vector_conditions = [f"message_id IN <vector_{i}>" for i in range(len(vector_queries))]
            sql = f"SELECT message_id FROM email WHERE {' AND '.join(vector_conditions)}"
        else:
            sql = "SELECT message_id FROM email"
        
        return {
            "sql": sql,
            "vector_query_list": vector_queries
        }
        
    except Exception as e:
        print(f"处理text_requirements时出错: {e}")
        # 返回默认查询
        return {
            "sql": "SELECT message_id FROM email WHERE message_id IN <vector_0>",
            "vector_query_list": [{"email_content": str(requirements)}]
        }
def process_single_pipeline(cands, rels, sql_cond, props, q1, q2, gold, pipeline_id):
    """处理单个pipeline的后续步骤（线程安全版本）"""
    print(f"[PIPELINE-{pipeline_id}] 开始处理 {len(cands)} 个候选项")
    
    try:
        
        final = filter_candidates(rels, props, cands)
        print(f"[PIPELINE-{pipeline_id}] filter_candidates 完成，找到 {len(final)} 个匹配项")
       
        text_sql_and_vector_query_list = text_requirements_to_sql_vector(props)
    
        return {
            "rel_requirements": rels,
            "text_requirements": props,
            "rel_sql": sql_cond,
            "text_sql": text_sql_and_vector_query_list["sql"],
            "vector_query_list": text_sql_and_vector_query_list["vector_query_list"],
            "raw_query": q1,
            "query": q2,
            "ground_truth": final,
            "gold_id": gold,
        }
    
        
    except Exception as e:
        print(f"[PIPELINE-{pipeline_id}] 处理过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        raise

def pipeline(maximum=120, minimum=5):
    """
    修改后的pipeline函数：
    1. 先生成足够的候选项（120个，最少5个）
    2. 分成3个并行处理，每个处理40个候选项
    3. 返回3个结果的列表
    """
    # 第一步：生成足够的候选项
    while True:
        rels = sample_relational_requirements()
        sql_cond = rel_reqs_to_sql(rels)
        print(f"[PIPELINE] 组合 SQL: SELECT message_id FROM email WHERE {sql_cond}")
        cursor.execute(f"SELECT message_id FROM email WHERE {sql_cond}")
        cands = [r[0] for r in cursor.fetchall()]
      

        if len(cands) <= maximum and len(cands) >= minimum:
            break
        elif len(cands) > maximum:
            cands = cands[:maximum]
            break
        else:
            print(f"[PIPELINE][WARN] 候选过少 ({len(cands)} < {minimum})，重新采样")
        # 挑选 gold 继续后续步骤
    gold = random.choice(cands)
    print(f"随机选 gold ID: {gold}")

    print(f"开始调用 extract_textual_requirements...")
    props = extract_textual_requirements(gold)
    print(f"extract_textual_requirements 完成")
        
    print(f"开始调用 compose_queries...")
    q1, q2 = compose_queries(rels, props)
    print(f"compose_queries 完成")

    print(f"开始调用 filter_candidates...")
    # 第二步：将候选项分成3组，每组约40个
    chunk_size = 40
    cand_chunks = []
    
    if len(cands) < 120:
        chunk_size = len(cands) // 3
        if chunk_size == 0:
            chunk_size = 1
    
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(cands))
        if start_idx < len(cands):
            chunk = cands[start_idx:end_idx]
            if chunk:
                cand_chunks.append(chunk)

    # 第三步：并行处理每组候选项
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交任务
        future_to_pipeline = {
            executor.submit(process_single_pipeline, chunk, rels, sql_cond, props, q1, q2, gold, i+1): i+1
            for i, chunk in enumerate(cand_chunks)
        }
        
        # 收集结果
        for future in as_completed(future_to_pipeline):
            try:
                result = future.result()
                results.append(result)
                print(f"ground_truth数量: {len(result.get('ground_truth', []))}")
            except Exception as exc:
                print(f"处理异常: {exc}")
                import traceback
                traceback.print_exc()

    print(f"共生成 {len(results)} 个结果")
    return results

if __name__ == "__main__":
    with open("pipeline_results_2300_1.jsonl", "a", encoding="utf-8") as f:
        for _ in range(800):
            print(f"\n=== 开始第 {_ + 1} 次迭代 ===")
            results = pipeline()  # 现在返回3个结果的列表
            
            # 处理每个并行pipeline的结果
            for result in results:
                if len(result["ground_truth"]) > 0:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(f"[MAIN] gold_id={result['gold_id']}, ground_truth数量={len(result['ground_truth'])}")
                else:
                    print(f"[MAIN]  Ground-truth，跳过保存")

            print(f"=== 第 {_ + 1} 次迭代完成，共保存 {len([r for r in results if len(r['ground_truth']) > 0])} 个有效结果 ===")

    print("[MAIN] 所有管道执行完毕")
    cursor.close()
    conn.close()