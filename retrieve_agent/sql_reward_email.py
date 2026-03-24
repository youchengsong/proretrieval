from dotenv import load_dotenv
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
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.output_parsers.json import SimpleJsonOutputParser

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

def genai(temperature=0.1, top_p=0.2):
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

# ---------- 工具：解析 inputs ----------
def parse_text_requirements(text_requirements: str) -> List[Dict[str, str]]:
    """text_requirements 是带 ```json ... ``` 的字符串，这里提取为 list[ {字段中文名: 内容} ]"""
    m = re.search(r"```json\s*(.*?)\s*```", text_requirements, flags=re.S)
    payload = m.group(1) if m else text_requirements
    return json.loads(payload)


CN2FIELD = {
    "发件人邮箱": "sender_email",
    "发件人姓名": "sender_name",
    "收件人列表": "recipient_list",
    "抄送列表": "cc_list",
    "密送列表": "bcc_list",
    "文件夹标签": "folder_labels",
    "附件列表": "attachment_list",
    "主题": "subject",
    "正文": "content",
}

ALLOWED_IDS = {"message_id", "folder_id", "attachment_id", "person_id"}
ALLOWED_VECTOR_FIELDS = {
    "sender_email","sender_name","recipient_list","cc_list","bcc_list",
    "folder_labels","attachment_list","subject","content",
    # person 表：
    # "contact_name","relationship","person_id"
}


class EmailSQLRewarder:
    """End-to-end wrapper that computes a reward for SQL + vector_query_list candidates.
    
    LLM client (gpt4o) is constructed *inside* this class. Resolution order:
      1) Azure AD (DefaultAzureCredential) → Azure OpenAI (deployment_name).
      2) Azure with API key (AZURE_OPENAI_API_KEY).
      3) OpenAI with API key (OPENAI_API_KEY), model fallback 'gpt-4o'.
    """

    SYSTEM_PROMPT = "你是严格的检索策略评审员，请根据给定的规则为候选 SQL+向量参数打分。只返回 JSON。"

    JUDGE_TMPL = """
# 关系需求（逐条满足）
{rel_requirements}

# 文本需求（字段→内容，允许同义改写但不可改变语义）
{text_requirements}

# 候选输出（SQL + vector_query_list）
{candidate_json}

评分维度（0~1，小数可用）：
- coverage_rel：是否覆盖关系需求（账户、日期），且无错误归属；
- coverage_text：是否覆盖文本属性（主题/正文/附件），允许同义但不可遗漏；
- field_mapping：语义→字段是否正确；
- keyword_correctness：关键词不臆造；
- time_format：日期筛选更推荐使用相对时间（如 date('now', ...)）或 LIKE/区间，而非死等号；
- deduplication：查询不会因笛卡尔积导致相同 ID 多次返回（如使用 EXISTS / DISTINCT / GROUP BY 规避）；
- overall：总体评分。

严格返回：
{{
  "coverage_rel": 0.0,
  "coverage_text": 0.0,
  "field_mapping": 0.0,
  "keyword_correctness": 0.0,
  "overall": 0.0,
  "notes": "一句话简述主要扣分点"
}}
"""

    def __init__(self, 
                 temperature: float = 0.2,
                 top_p: float = 0.2,
                 azure_endpoint: Optional[str] = None,
                 deployment_name: Optional[str] = None,
                 api_version: Optional[str] = None,
                 model_fallback: str = "gpt-4o",
                 llm_enabled: bool = True):
        # Runtime config (env as default)
        self.temperature = temperature
        self.top_p = top_p
        self.azure_endpoint = azure_endpoint or os.getenv("LLM_AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("LLM_AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt4o"
        self.api_version = api_version or os.getenv("LLM_AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"
        self.model_fallback = model_fallback
        self.llm_enabled = llm_enabled