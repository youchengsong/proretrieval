import asyncio
import json
import re
import traceback

from retrieve_env import execute_sql, search_mail
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_google_vertexai import ChatVertexAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

def gpt4o(temperature=0.3, top_p=0.2):
    return AzureChatOpenAI(
    # model_name="gpt-4o-2024-08-06",
    openai_api_version="2024-08-01-preview",
    azure_endpoint="https://loox-eastus.openai.azure.com/",
    deployment_name="gpt4o",
    # deployment_name="gpt4o-0513",
    azure_ad_token_provider=token_provider,
    temperature=temperature,
    top_p=top_p,
)

def extract_sql_vector_block(log_text: str) -> str:
    """
    在 TaskRunner 日志中查找 <answer>...</answer> 标签包围的 JSON，
    并返回形如 {"sql": ..., "vector_query_list": ...} 的字符串块。
    若找不到则返回空字符串 ""。
    """
    # ① 先定位 <answer>...</answer> 块（非贪婪）
    m = re.search(r"<answer>(.*?)</answer>", log_text, flags=re.S)
    if not m:
        print("No <answer>...</answer> block found.")
        return ""

    answer_block = m.group(1)  # 只取标签内部文本

    # ② 在 answer_block 内找到第一个 “{” 并配对 “}”
    start = answer_block.find("{")
    if start == -1:
        print("No JSON object found inside <answer> block.")
        return ""

    brace_depth = 0
    end = None
    for idx, ch in enumerate(answer_block[start:], start=start):
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end = idx + 1
                break

    if end is None:
        print("Unmatched braces in JSON block.")
        return ""

    json_block = answer_block[start:end]
    return json_block

async def main():
    data = []
    with open('/workspace/yunhai/rl-date-gen/pipeline_results_v5.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    res = []
    empty = []
    error = []
    with open("/workspace/yunhai/rl-date-gen/passed_result_v5.jsonl", 'w', encoding='utf-8') as f_passed, \
            open("/workspace/yunhai/rl-date-gen/empty_result_v5.jsonl", 'w', encoding='utf-8') as f_empty, \
            open("/workspace/yunhai/rl-date-gen/error_result_v5.jsonl", 'w', encoding='utf-8') as f_error:
        for item in data:
            try:
                print(item)
                query = item.get('raw_query', None)
                template = """
# Role Description
- 你是一位资深 AI 文件检索系统工程师，擅长将自然语言查询请求转化为带有向量工具查询占位符的 SQL 语句，用于查询结构化与非结构化混合字段构成的邮件数据。

# Responsibilities
- 针对用户的自然语言查询需求，生成如下内容：
  1. 一条带有 <vector_x> 占位符的 SQL 查询语句；
  2. 与之对应的向量查询字段及内容组成的参数列表 vector_query_list。
- 所有涉及非结构化文本字段（如姓名、主题、邮件内容、附件名称与内容、标签名称）必须使用向量工具匹配，不能直接参与 SQL LIKE 等操作。
- 最终只返回以下 ID 列：  
  - `message_id`（email 表）  
  - `folder_id`（folder_labels）  
  - `attachment_id`（attachment_list）  
  - `person_id`（person 表）  
  SQL 中不得返回其他字段。

# 数据表结构说明（仅支持在一张表中检索）

## email 表 （邮件信息，仅支持独立检索）

- `message_id`            TEXT     -- 邮件唯一 ID  
- `thread_id`             TEXT     -- 邮件所在Thread ID  
- `account_email`         TEXT     -- 邮箱所属账户（结构化，可直接用 SQL 条件匹配）  
- `sender_email`          TEXT     -- 发件人邮箱地址（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `sender_name`           TEXT     -- 发件人姓名（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `recipient_list`        TEXT     -- 收件人列表 [{{"email":"xxx","name":"xxx"}}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `cc_list`               TEXT     -- 抄送人列表 [{{"email":"xxx","name":"xxx"}}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `bcc_list`              TEXT     -- 密送人列表 [{{"email":"xxx","name":"xxx"}}]（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `received_date`         TEXT     -- 邮件收件时间（结构化，可用 `LIKE '2025%'` 等 SQL 条件匹配）  
- `is_draft`              INTEGER  -- 是否是草稿（结构化，可直接通过 SQL 条件匹配，如 `is_draft = 1` 表示是草稿）  
- `draft_created_date`    TEXT     -- 草稿创建时间（仅适用于草稿邮件，结构化可匹配）  
- `draft_modified_date`   TEXT     -- 草稿修改时间（仅适用于草稿邮件，结构化可匹配）  
- `subject`               TEXT     -- 邮件主题（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `email_content`         TEXT     -- 邮件正文内容（非结构化，需要使用向量检索工具，返回符合语义的 message_id 列表）  
- `is_read`               INTEGER  -- 是否已读（结构化，可直接通过 SQL 条件匹配，如 `is_read = 0` 或 `is_read = 1`）  
- `is_starred`            INTEGER  -- 是否星标（结构化，可直接通过 SQL 条件匹配，如 `is_starred = 1` 表示已星标）  
- `is_archived`           INTEGER  -- 是否归档（结构化，可直接通过 SQL 条件匹配，如 `is_archived = 0` 或 `is_archived = 1`）  
- `folder_labels`         TEXT     -- 标签列表 [{{"id":"xxx","name":"xxx"}}]（非结构化，需要使用向量检索工具，返回对应的 folder_id 列表）  
- `thread_message_count`  INTEGER  -- 邮件所在对话数量（结构化，可直接匹配）  
- `attachment_list`       TEXT     -- 附件列表 [{{"id":"xxx","name":"xxx","type":"xxx","size":xxx,"content":"xxx"}}]（非结构化，需要使用向量检索工具，返回符合语义的 attachment_id 列表）  

## person 表（联系人信息，仅支持独立检索）

- `person_id`         TEXT     -- 联系人唯一 ID（主键）以及联系人邮箱地址（非结构化，需要使用向量检索工具，返回符合语义的 person_id 列表）  
- `contact_name`      TEXT     -- 联系人姓名（非结构化，需要使用向量检索工具，返回符合语义的 person_id 列表）  
- `email_count`       INTEGER  -- 与该联系人的往来邮件数量（结构化，可用于 SQL 条件过滤）  
- `relationship`      TEXT     -- 与该联系人的关系（如"同事"、"家人"等）（非结构化，需使用向量检索工具，返回符合语义的 person_id 列表）  

## 不支持混表查询（仅支持一次查询 email 表或 person 表）

# 工具使用规范
- 向量匹配字段必须以 `<vector_0>` 至 `<vector_9>` 占位符形式嵌入 SQL。  
- 当需要对多个非结构化字段进行向量检索时，依次使用 `<vector_0>`、`<vector_1>`、`<vector_2>` …，按字段在 SQL 中出现的顺序分配占位符，最多支持 10 个向量查询。  
- 所有向量字段仅通过逻辑筛选接入，具体规则：  
  - 对于返回`message_id`的字段（如`subject`, `sender_name`, `recipient_list`, `cc_list`, `bcc_list`, `email_content`），SQL中应使用`message_id IN <vector_x>`；  
  - 对于返回其他ID的字段（如`folder_labels`返回`folder_id`，`attachment_list`返回`attachment_id`），SQL中应使用`json_extract(...) IN <vector_x>`；  
  - 对于`person`表，非结构化字段返回`person_id`，SQL中应使用`person_id IN <vector_x>`。  
- `recipient_list`、`cc_list`、`bcc_list`、`folder_labels`、`attachment_list` 均为 JSON 列，匹配时须借助 `json_each()` 与 `json_extract()`。  
- 对于日期相关模糊查询（如"最近"），应使用相对日期函数（如`date('now', '-7 day')`）进行动态计算，避免硬编码。  

# 输出格式
仅输出一个 JSON 对象，使用 <answer></answer> 包裹，不要任何额外注释或说明。  
结构如下：  
<answer>  
{{  
  "sql": "SQL 查询语句（带有 <vector_x> 占位符）",  
  "vector_query_list": [  
    {{ "字段名1": "向量查询内容" }},  
    {{ "字段名2": "向量查询内容" }}  
  ]  
}}  
</answer>

# 模板示例
（模版中的 query 为输入的 query，生成阶段不输出 query）

- query: "帮我找 2025 年标星邮件中，主题包含预算且附件内容包含合同的邮件"  
<answer>  
{{  
  "sql": "SELECT DISTINCT message_id FROM email, json_each(email.attachment_list) WHERE received_date LIKE '2025%' AND is_starred = 1 AND message_id IN <vector_0> AND json_extract(json_each.value, '$.id') IN <vector_1>",  
  "vector_query_list": [  
    {{ "subject": "预算" }},  
    {{ "attachment_list": "合同" }}  
  ]  
}}  
</answer>

{query}
"""
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | gpt4o() | StrOutputParser()
                res = chain.invoke({
                    "query": query
                })
                if not query:
                    print("❌ 查询语句为空，跳过此条记录")
                    continue
                
                sql = extract_sql_vector_block(res)

                print(f'✅ Extracted SQL: {sql}')
                
                res1 = execute_sql(str(sql), 10)
                print(res1)
                item['stmt'] = res1.get('stmt', None)
                item['monc_sql'] = sql
                if not res1:
                    print("❌ Query 执行失败，跳过此条记录")
                    error.append(item)
                    line = json.dumps(item, ensure_ascii=False)
                    f_error.write(line + '\n')
                    continue
                elif not res1['result']:
                    print("❌ Query 没有找到任何结果")
                    empty.append(item)
                    line = json.dumps(item, ensure_ascii=False)
                    f_empty.write(line + '\n')
                    continue
                else:
                    res1 = [ it['id'] for it in res1['result']]
                if item['gold_id'] in res1:
                    acc = 1.0
                    line = json.dumps(item, ensure_ascii=False)
                    f_passed.write(line + '\n')
                else:
                    acc = 0.0   
                    line = json.dumps(item, ensure_ascii=False)
                    f_empty.write(line + '\n')

            except Exception as e:
                print("Error processing item:", item)
                traceback.print_exc()
                continue

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())