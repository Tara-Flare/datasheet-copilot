
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.schemas.parameter import ExtractionResult
from app.extraction.prompts import SYSTEM_PROMPT


load_dotenv()


class ParameterExtractor:
    """基于 LLM 的 Datasheet 参数提取器。"""

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")
        base_url = os.getenv("LLM_BASE_URL")

        if not api_key:
            raise ValueError("缺少 LLM_API_KEY")

        if not model:
            raise ValueError("缺少 LLM_MODEL")

        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=60.0,
            max_retries=2
        )

    def extract(
        self,
        tables: list[dict]
    ) -> dict:
        """
        从 Docling 表格数据中提取电气参数。
        """

        if not tables:
            return ExtractionResult(
                component_model="",
                manufacturer="",
                parameters=[]
            ).model_dump()

        table_data = json.dumps(
            tables,
            ensure_ascii=False
        )

        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        "请从以下 Datasheet 表格中"
                        "提取电气参数：\n\n"
                        + table_data
                    )
                }
            ],
            response_format=ExtractionResult
        )

        message = response.choices[0].message

        if message.refusal:
            raise ValueError(
                f"模型拒绝提取: {message.refusal}"
            )

        result = message.parsed

        if result is None:
            raise ValueError(
                "模型未返回有效的结构化结果"
            )

        return result.model_dump()


    
    def retry_parameters(
        self,
        failed_parameters: list[dict],
        tables: list[dict]
    ) -> dict:
        """
        根据错误字段和反馈重新提取参数。

        只向模型提供错误字段及相关表格。
        """
        import json

        if not failed_parameters:
            return {
                "parameters": []
            }

        # 获取失败字段引用的表格
        table_ids = {
            p.get("evidence", {}).get("table_id")
            for p in failed_parameters
        }

        candidate_tables = [
            table
            for table in tables
            if table["table_id"] in table_ids
        ]

        retry_input = {
            "failed_parameters": failed_parameters,
            "source_tables": candidate_tables
        }

        prompt = """
请根据来源表格重新提取失败的参数。

要求：
1. 只纠正 failed_parameters 中列出的字段。
2. 认真检查 validation_errors。
3. 根据 Min、Typ、Max 列确定参数类型。
4. 必须保留真实 table_id、row_index、page。
5. 不允许编造来源。
6. 无法确定时直接返回空参数列表。
7. 不要输出无关参数。
"""

        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        retry_input,
                        ensure_ascii=False
                    )
                }
            ],
            response_format=ExtractionResult
        )

        message = response.choices[0].message

        if message.refusal or message.parsed is None:
            raise ValueError(
                "参数纠错未返回有效结果"
            )

        return message.parsed.model_dump()

