
import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from app.schemas.parameter import ExtractionResult
from app.extraction.prompts import (
    FLAT_FAMILY_COVERAGE,
    LOGICAL_INPUT_INSTRUCTION,
    LOGICAL_INPUT_INSTRUCTION_BASE,
    LOGICAL_RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
)
from app.extraction.input_mode import (
    build_extraction_input,
    is_logical_table,
    merge_logical_tables,
    resolve_extraction_input_mode,
)
from app.extraction.logical_source import (
    canonicalize_parameters,
)


load_dotenv()

logger = logging.getLogger(__name__)


class OutputTruncationError(ValueError):
    """模型输出被截断（输出过长或上下文超限）。"""


EXTRACT_INSTRUCTION = """
请从候选表格中提取 LDO 电气参数。

要求：
1. 只根据表格中的内容提取。
2. 不允许编造任何参数。
3. Min、Typ、Max 分别输出。
4. 必须保留测试条件。
5. table_id 必须引用输入中的表格 ID。
6. row_index 必须引用输入表格 row_indices 数组中与 rows 对应位置的值
   （即该行在原始表格中的行号，从 0 开始）。
   不得使用分块内从 0 重新编号的局部行号。
7. page 必须来自对应表格的 page_numbers。
8. source_text 应复制来源行中的原始文字。
9. 无法确定的参数直接跳过。
10. 不要将绝对最大额定值误认为推荐工作条件。
"""


RETRY_INSTRUCTION = """
你正在执行参数纠错任务。

输入中的 failed_parameters 是校验失败的参数。
每个参数都附带了：
- name / value / unit / value_type / test_conditions
- validation_errors：校验失败原因
- table_id / row_index / page：原始来源定位
- columns：原始表格列名
- source_row：该参数所在的原始表格行内容

要求：
1. 仅纠正 failed_parameters 中列出的参数。
2. 根据 source_row 与 columns 重新确定正确的
   value、unit、value_type 和 test_conditions。
3. 检查数值是否位于正确的 Min、Typ、Max 列。
4. 在 parameters 数组中输出纠正后的参数，
   每个参数必须包含完整字段：
   name、value、unit、value_type、test_conditions、evidence。
5. evidence.table_id、evidence.row_index、evidence.page
   必须与输入完全一致，不得修改。
6. evidence.source_text 应从 source_row 中摘录原文。
7. 不允许编造来源、数值、单位或测试条件。
8. 无法纠正的参数直接跳过，不要输出。
"""


class DeepSeekExtractor:
    """DeepSeek Datasheet 参数提取适配器。"""

    def __init__(
        self,
        *,
        input_mode=None,
        logical_rows=None,
        raw_structure=None,
        logical_table_ids=None,
        include_needs_review=False,
        family_coverage=None,
    ):
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv(
            "LLM_BASE_URL",
            "https://api.deepseek.com"
        )
        model = os.getenv(
            "LLM_MODEL",
            "deepseek-flash"
        )

        if not api_key:
            raise ValueError(
                "缺少 LLM_API_KEY，请检查 .env"
            )

        self.model = model

        # 阶段 B3.2a：提取输入模式。
        # flat（默认）保持旧行为；logical 使用 LogicalTableAdapter。
        self.input_mode = resolve_extraction_input_mode(
            input_mode
        )
        self.logical_rows = logical_rows
        self.raw_structure = raw_structure
        self.logical_table_ids = logical_table_ids
        self.include_needs_review = bool(
            include_needs_review
        )

        # B3.6 A/B：低频参数族 Prompt 覆盖开关。
        # 默认关闭，保证 B3.5c 基线可复现；
        # B3.6 各实验臂通过 PROMPT_FAMILY_COVERAGE=1 开启。
        if family_coverage is None:
            family_coverage = os.getenv(
                "PROMPT_FAMILY_COVERAGE", "0"
            ).strip().lower() in (
                "1", "true", "yes", "on",
            )

        self.family_coverage = bool(family_coverage)

        # MVP 生产默认固定 temperature=0.0；
        # 仍可通过 LLM_TEMPERATURE 显式覆盖。
        try:
            self.temperature = float(
                os.getenv("LLM_TEMPERATURE", "0.0")
            )
        except (TypeError, ValueError):
            self.temperature = 0.0

        # 单次响应的输出 Token 上限。
        # 可通过 LLM_MAX_TOKENS 环境变量覆盖。
        try:
            self.max_tokens = int(
                os.getenv("LLM_MAX_TOKENS", "6000")
            )
        except (TypeError, ValueError):
            self.max_tokens = 6000

        # 行级分块大小：每块最多处理的行数。
        try:
            self.rows_per_chunk = int(
                os.getenv("LLM_ROWS_PER_CHUNK", "10")
            )
        except (TypeError, ValueError):
            self.rows_per_chunk = 10

        if self.rows_per_chunk < 1:
            self.rows_per_chunk = 10

        # 截断自动二分拆分的最大递归深度。
        try:
            self.max_split_depth = int(
                os.getenv("LLM_MAX_SPLIT_DEPTH", "6")
            )
        except (TypeError, ValueError):
            self.max_split_depth = 6

        if self.max_split_depth < 1:
            self.max_split_depth = 6

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90.0,
            max_retries=2,
        )

        # 记录本次运行的模型 Token 用量（多次调用累计）
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # B3.2b 实验用：逐次保留原始 LLM 输出。
        # 只记录响应，不改变提取逻辑。
        self.raw_responses = []

    def _record_raw_response(self, record):
        """保留原始响应，供 B3.2b A/B 实验审计。"""

        if not hasattr(self, "raw_responses"):
            self.raw_responses = []

        self.raw_responses.append(record)

    def _request_json(
        self,
        payload: dict,
        instruction: str
    ) -> dict:
        """调用 DeepSeek JSON Output 接口。"""

        system_prompt = f"""
{SYSTEM_PROMPT}

{instruction}

必须只返回合法的 JSON 对象。

JSON 输出示例：
{{
  "component_model": "",
  "manufacturer": "",
  "parameters": []
}}

parameters 中的每个参数必须符合以下规则：
- name: 字符串
- value: 数字
- unit: 字符串
- value_type: min、typ、max 或 nominal
- test_conditions: 字符串
- evidence.table_id: 整数
- evidence.row_index: 整数，从 0 开始
- evidence.page: 整数，从 1 开始
- evidence.source_text: 原文摘录

不要输出 Markdown 代码块。
不要输出 JSON 以外的解释。
"""

        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={
                    "type": "json_object"
                },
                max_tokens=self.max_tokens,
                temperature=getattr(
                    self, "temperature", 0.0
                ),
            )

            usage = response.usage
            prompt_tokens = (
                usage.prompt_tokens or 0
            ) if usage else 0
            completion_tokens = (
                usage.completion_tokens or 0
            ) if usage else 0
            reasoning_tokens = None

            if usage:
                reasoning_tokens = getattr(
                    usage, "reasoning_tokens", None
                )

                if reasoning_tokens is None:
                    details = getattr(
                        usage,
                        "completion_tokens_details",
                        None,
                    )

                    if details is not None:
                        reasoning_tokens = getattr(
                            details,
                            "reasoning_tokens",
                            None,
                        )

            # 多次 API 调用的 Token 累计
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            choice = response.choices[0]
            content = choice.message.content

            logger.info(
                "DeepSeek 请求完成: finish_reason=%s, "
                "prompt_tokens=%s, completion_tokens=%s, "
                "reasoning_tokens=%s, max_tokens=%s",
                choice.finish_reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                self.max_tokens,
            )

            raw_record = {
                "attempt": attempt,
                "finish_reason": choice.finish_reason,
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "max_tokens": getattr(
                    self, "max_tokens", None
                ),
            }

            if choice.finish_reason == "length":
                raw_record["error"] = "output_truncated"
                self._record_raw_response(raw_record)

                raise OutputTruncationError(
                    "模型输出被截断，请减少输入表格数量"
                )

            if not content or not content.strip():
                raw_record["error"] = "empty_content"
                self._record_raw_response(raw_record)

                if attempt == 0:
                    continue

                raise ValueError(
                    "DeepSeek 返回了空 JSON 内容"
                )

            try:
                parsed = json.loads(content)

                # 结构校验
                result = ExtractionResult.model_validate(
                    parsed
                )

                raw_record["validated"] = True
                self._record_raw_response(raw_record)

                # 新 Evidence 可选字段在 Flat 模式下保持
                # 与旧输出完全相同的形状。
                return result.model_dump(exclude_none=True)

            except (
                json.JSONDecodeError,
                ValidationError
            ) as exc:
                raw_record["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_raw_response(raw_record)

                if attempt == 1:
                    raise ValueError(
                        f"模型结构化输出校验失败: {exc}"
                    ) from exc

        raise RuntimeError("LLM 提取失败")

    def _build_table_payload(
        self,
        table: dict,
        rows: list,
        row_indices: list,
        record_positions: list | None = None,
    ) -> dict:
        """构造带原始行号映射的表格载荷。

        Flat 模式保持旧载荷字段不变；
        Logical 模式额外附带 logical_row_ids 与完整来源数组，
        供程序化来源映射使用。
        """

        payload = {
            "table_id": table.get("table_id"),
            "page_numbers": table.get("page_numbers", []),
            "columns": table.get("columns", []),
            "rows": rows,
            "row_indices": row_indices,
            "row_count": len(rows),
            "column_count": table.get(
                "column_count",
                len(table.get("columns", [])),
            ),
        }

        if is_logical_table(table):
            records = table.get("records") or []

            if record_positions is None:
                record_positions = list(range(len(rows)))

            chunk_records = [
                records[position]
                for position in record_positions
                if 0 <= position < len(records)
            ]

            payload["input_mode"] = "logical"
            payload["logical_row_ids"] = [
                record.get("logical_row_id")
                for record in chunk_records
            ]
            payload["source_row_indices"] = [
                list(record.get("source_row_indices") or [])
                for record in chunk_records
            ]
            payload["source_cell_ids"] = [
                list(record.get("source_cell_ids") or [])
                for record in chunk_records
            ]

        return payload

    def _merge_results(
        self,
        results: list[dict],
    ) -> dict:
        """合并多个批次的提取结果。"""

        merged = {
            "component_model": "",
            "manufacturer": "",
            "parameters": [],
        }

        for result in results:
            if (
                not merged["component_model"]
                and result.get("component_model")
            ):
                merged["component_model"] = (
                    result["component_model"]
                )

            if (
                not merged["manufacturer"]
                and result.get("manufacturer")
            ):
                merged["manufacturer"] = (
                    result["manufacturer"]
                )

            merged["parameters"].extend(
                result.get("parameters", [])
            )

        return merged

    def _split_rows(
        self,
        rows: list,
        row_indices: list | None = None,
    ) -> list[tuple[list, list, list]]:
        """按行边界将表格拆分为多个行块。

        每一块返回:
        - 行内容；
        - 该行对应的原始 row_index（Logical 模式使用
          table["row_indices"]，Flat 模式保持从 0 递增）；
        - 行在表格中的位置（用于对齐 logical records）。

        保证 evidence.row_index 始终指向原始行，
        而不是分块内重新编号的局部行号。
        """

        if row_indices is None:
            indices = list(range(len(rows)))
        else:
            indices = list(row_indices)

            if len(indices) != len(rows):
                raise ValueError(
                    "row_indices 与 rows 长度不一致: "
                    f"{len(indices)} != {len(rows)}"
                )

        positions = list(range(len(rows)))
        chunks = []

        for start in range(
            0, len(rows), self.rows_per_chunk
        ):
            end = start + self.rows_per_chunk

            chunks.append((
                rows[start:end],
                indices[start:end],
                positions[start:end],
            ))

        return chunks

    def _extract_chunk(
        self,
        table: dict,
        rows: list,
        row_indices: list,
        depth: int = 0,
        record_positions: list | None = None,
    ) -> dict:
        """提取一个行块。

        若返回 finish_reason=length（截断），
        按原始行边界二分后递归处理两个子块，
        直到成功或达到最大拆分深度。
        """

        table_id = table.get("table_id")
        logical = is_logical_table(table)

        payload = self._build_table_payload(
            table,
            rows,
            row_indices,
            record_positions=record_positions,
        )

        logger.info(
            "提取行块: table_id=%s, 深度=%s, 行数=%s, "
            "输入字符数=%s, 起始原始行号=%s",
            table_id,
            depth,
            len(rows),
            len(json.dumps(payload, ensure_ascii=False)),
            row_indices[0] if row_indices else None,
        )

        instruction = EXTRACT_INSTRUCTION
        family_coverage = getattr(
            self, "family_coverage", False
        )

        if logical:
            instruction += (
                LOGICAL_INPUT_INSTRUCTION
                if family_coverage
                else LOGICAL_INPUT_INSTRUCTION_BASE
            )
        elif family_coverage:
            instruction += FLAT_FAMILY_COVERAGE

        try:
            return self._request_json(
                payload={"tables": [payload]},
                instruction=instruction,
            )
        except OutputTruncationError:
            # 单行仍然截断：无法继续拆分
            if len(rows) <= 1:
                raise OutputTruncationError(
                    f"单行仍被截断: table_id={table_id}, "
                    f"row_index="
                    f"{row_indices[0] if row_indices else None}"
                )

            if depth >= self.max_split_depth:
                raise OutputTruncationError(
                    f"已达到最大拆分深度 {self.max_split_depth}: "
                    f"table_id={table_id}, 行数={len(rows)}, "
                    f"row_indices={row_indices}"
                )

            mid = len(rows) // 2

            left_positions = (
                record_positions[:mid]
                if record_positions is not None
                else None
            )
            right_positions = (
                record_positions[mid:]
                if record_positions is not None
                else None
            )

            logger.warning(
                "检测到截断，按行二分: table_id=%s, "
                "行数=%s -> %s + %s",
                table_id,
                len(rows),
                mid,
                len(rows) - mid,
            )

            left = self._extract_chunk(
                table,
                rows[:mid],
                row_indices[:mid],
                depth + 1,
                record_positions=left_positions,
            )

            right = self._extract_chunk(
                table,
                rows[mid:],
                row_indices[mid:],
                depth + 1,
                record_positions=right_positions,
            )

            return self._merge_results([left, right])

    def _prepare_tables(self, tables, logical_input=None):
        """按输入模式装配 Flat / Logical 表格。

        - 已经包含 Logical 表格时不再重复适配；
        - 显式传入 adapter 输出时直接合并；
        - 构造时提供 logical_rows 时在本方法内适配；
        - 没有可用 Logical 来源时安全回退 Flat，绝不丢表。
        """

        if any(
            is_logical_table(table) for table in tables
        ):
            return tables

        if logical_input is not None:
            return merge_logical_tables(
                tables, logical_input
            )["tables"]

        logical_rows = getattr(
            self, "logical_rows", None
        )

        mode = getattr(self, "input_mode", "flat")

        if mode == "logical" and isinstance(
            logical_rows, dict
        ):
            return build_extraction_input(
                tables,
                mode="logical",
                logical_rows=logical_rows,
                raw_structure=getattr(
                    self, "raw_structure", None
                ),
                table_ids=getattr(
                    self, "logical_table_ids", None
                ),
                include_needs_review=getattr(
                    self, "include_needs_review", False
                ),
            )["tables"]

        if mode == "logical":
            logger.warning(
                "EXTRACTION_INPUT_MODE=logical，但未提供 "
                "logical_rows；本次继续使用 Flat 表格"
            )

        return tables

    def _canonicalize_source_fields(
        self,
        parameters,
        tables,
    ):
        """用程序核对并重写 Logical 来源字段。"""

        logical_tables = [
            table for table in tables
            if is_logical_table(table)
        ]

        if not logical_tables:
            return parameters

        return canonicalize_parameters(
            parameters, logical_tables
        )

    def extract(
        self,
        tables: list[dict],
        logical_input=None,
    ) -> dict:
        """从候选表格中提取参数。

        逐表处理，每张表按行分块调用 LLM；
        截断时自动二分降级，最后合并结果。

        Logical 模式下，LLM 输出的来源字段会在合并后
        由程序按 canonical record 重新核对与重写。
        """

        if not tables:
            return {
                "component_model": "",
                "manufacturer": "",
                "parameters": [],
            }

        tables = self._prepare_tables(
            tables, logical_input=logical_input
        )

        merged = {
            "component_model": "",
            "manufacturer": "",
            "parameters": [],
        }

        total_tables = len(tables)
        failures = []

        for table_index, table in enumerate(
            tables,
            start=1,
        ):
            table_id = table.get("table_id")
            rows = table.get("rows", [])
            logical = is_logical_table(table)

            logger.info(
                "提取表格: 第 %s/%s 张, table_id=%s, "
                "输入模式=%s, 行数=%s, 输入字符数=%s",
                table_index,
                total_tables,
                table_id,
                "logical" if logical else "flat",
                len(rows),
                len(json.dumps(table, ensure_ascii=False)),
            )

            row_indices = (
                table.get("row_indices")
                if logical
                else None
            )

            for (
                chunk_rows,
                chunk_indices,
                chunk_positions,
            ) in self._split_rows(
                rows, row_indices=row_indices
            ):
                try:
                    result = self._extract_chunk(
                        table,
                        chunk_rows,
                        chunk_indices,
                        record_positions=chunk_positions,
                    )
                except Exception as exc:
                    logger.error(
                        "提取行块失败: table_id=%s, "
                        "起始原始行号=%s, 异常类型=%s: %s",
                        table_id,
                        chunk_indices[0]
                        if chunk_indices else None,
                        type(exc).__name__,
                        exc,
                    )
                    failures.append({
                        "table_id": table_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    })
                    continue

                merged = self._merge_results([
                    merged,
                    result,
                ])

                logger.info(
                    "提取行块完成: table_id=%s, "
                    "本块参数=%s, 累计参数=%s",
                    table_id,
                    len(result.get("parameters", [])),
                    len(merged["parameters"]),
                )

        if failures:
            failed_ids = sorted({
                failure["table_id"]
                for failure in failures
            })

            exception_types = [
                failure["exception_type"]
                for failure in failures
            ]

            raise ValueError(
                f"参数提取失败: {len(failures)} 个行块"
                f"提取失败, 失败表格 table_id={failed_ids}, "
                f"异常类型={exception_types}"
            )

        merged["parameters"] = (
            self._canonicalize_source_fields(
                merged["parameters"], tables
            )
        )

        return merged

    def _logical_record_for_evidence(
        self,
        table,
        evidence,
    ):
        """按 logical_row_id 或唯一 row_index 定位逻辑行。

        多条逻辑行共享同一 row_index 时返回 None，
        不允许猜测。
        """

        records = table.get("records") or []

        logical_row_id = evidence.get("logical_row_id")

        if logical_row_id:
            matches = [
                (position, record)
                for position, record in enumerate(records)
                if record.get("logical_row_id")
                == logical_row_id
            ]

            if len(matches) == 1:
                return matches[0]

            return None

        row_index = evidence.get("row_index")

        matches = [
            (position, record)
            for position, record in enumerate(records)
            if record.get("primary_row_index") == row_index
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    def _build_correction_items(
        self,
        failed_parameters: list[dict],
        tables: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """把失败参数转换为可定位来源的纠错项。

        返回 (可定位项列表, 无法定位参数列表)。
        无法定位的参数不会发送给 LLM，
        交由现有审核机制处理。

        Logical 表格必须按 canonical logical row 定位；
        绝不把逻辑行位置直接当作原始 row_index。
        """

        table_map = {
            table.get("table_id"): table
            for table in tables
        }

        items = []
        unresolvable = []

        for param in failed_parameters:
            evidence = param.get("evidence") or {}
            table_id = evidence.get("table_id")
            row_index = evidence.get("row_index")
            page = evidence.get("page")

            table = table_map.get(table_id)

            if table is None:
                unresolvable.append(param)
                continue

            logical = is_logical_table(table)

            if logical:
                match = self._logical_record_for_evidence(
                    table, evidence
                )

                if match is None:
                    unresolvable.append(param)
                    continue

                position, record = match
                canonical_row_index = record.get(
                    "primary_row_index"
                )
                canonical_page = record.get("page")
                rows = table.get("rows", [])

                if (
                    not 0 <= position < len(rows)
                    or canonical_page
                    not in table.get("page_numbers", [])
                ):
                    unresolvable.append(param)
                    continue

                items.append({
                    "name": param.get("name"),
                    "value": param.get("value"),
                    "unit": param.get("unit"),
                    "value_type": param.get(
                        "value_type"
                    ),
                    "test_conditions": param.get(
                        "test_conditions", ""
                    ),
                    "validation_errors": param.get(
                        "validation_errors", []
                    ),
                    "table_id": table_id,
                    "row_index": canonical_row_index,
                    "page": canonical_page,
                    "logical_row_id": record.get(
                        "logical_row_id"
                    ),
                    "source_row_indices": list(
                        record.get(
                            "source_row_indices"
                        ) or []
                    ),
                    "columns": table.get("columns", []),
                    "source_row": rows[position],
                })
                continue

            rows = table.get("rows", [])

            if (
                not isinstance(row_index, int)
                or isinstance(row_index, bool)
                or not 0 <= row_index < len(rows)
            ):
                unresolvable.append(param)
                continue

            if page not in table.get("page_numbers", []):
                unresolvable.append(param)
                continue

            items.append({
                "name": param.get("name"),
                "value": param.get("value"),
                "unit": param.get("unit"),
                "value_type": param.get("value_type"),
                "test_conditions": param.get(
                    "test_conditions", ""
                ),
                "validation_errors": param.get(
                    "validation_errors", []
                ),
                "table_id": table_id,
                "row_index": row_index,
                "page": page,
                "columns": table.get("columns", []),
                "source_row": rows[row_index],
            })

        return items, unresolvable

    def _correct_items(
        self,
        items: list[dict],
        depth: int = 0,
    ) -> dict:
        """对一批纠错项调用 LLM，截断时按项二分递归。"""

        if not items:
            return {
                "component_model": "",
                "manufacturer": "",
                "parameters": [],
            }

        table_ids = sorted({
            item.get("table_id")
            for item in items
        })

        logger.info(
            "纠错批次: 参数数量=%s, 深度=%s, table_ids=%s",
            len(items),
            depth,
            table_ids,
        )

        instruction = RETRY_INSTRUCTION

        if any(item.get("logical_row_id") for item in items):
            instruction += LOGICAL_RETRY_INSTRUCTION

        try:
            return self._request_json(
                payload={"failed_parameters": items},
                instruction=instruction,
            )
        except OutputTruncationError:
            if len(items) <= 1:
                first = items[0]
                raise OutputTruncationError(
                    f"单个参数纠错仍被截断: "
                    f"name={first.get('name')}, "
                    f"table_id={first.get('table_id')}, "
                    f"row_index={first.get('row_index')}"
                )

            if depth >= self.max_split_depth:
                raise OutputTruncationError(
                    f"纠错达到最大拆分深度 "
                    f"{self.max_split_depth}: "
                    f"参数数量={len(items)}"
                )

            mid = len(items) // 2

            logger.warning(
                "纠错截断，按参数二分: %s -> %s + %s",
                len(items),
                mid,
                len(items) - mid,
            )

            left = self._correct_items(
                items[:mid],
                depth + 1,
            )

            right = self._correct_items(
                items[mid:],
                depth + 1,
            )

            return self._merge_results([left, right])

    def _restore_evidence(
        self,
        corrected_params: list[dict],
        failed_parameters: list[dict],
    ) -> list[dict]:
        """将纠错结果的证据恢复到原始来源定位。

        强制保留原始 table_id、row_index、page，
        避免模型篡改或漂移。
        """

        by_key = {}
        by_logical_id = {}

        for param in failed_parameters:
            key = (
                param.get("name"),
                param.get("value_type"),
                param.get("test_conditions"),
            )
            by_key.setdefault(key, []).append(param)

            logical_row_id = (
                param.get("evidence") or {}
            ).get("logical_row_id")

            if logical_row_id:
                by_logical_id.setdefault(
                    logical_row_id, []
                ).append(param)

        restored = []

        for corrected in corrected_params:
            key = (
                corrected.get("name"),
                corrected.get("value_type"),
                corrected.get("test_conditions"),
            )

            corrected_evidence = dict(
                corrected.get("evidence") or {}
            )
            corrected_logical_id = corrected_evidence.get(
                "logical_row_id"
            )

            matches = by_key.get(key, [])

            original = None
            corrected_table_id = corrected_evidence.get(
                "table_id"
            )

            # Logical 模式下 logical_row_id 是更强的匹配键。
            if corrected_logical_id:
                logical_matches = by_logical_id.get(
                    corrected_logical_id, []
                )

                if len(logical_matches) == 1:
                    original = logical_matches[0]

            if original is None:
                for match in matches:
                    match_table_id = (
                        match.get("evidence") or {}
                    ).get("table_id")

                    if match_table_id == corrected_table_id:
                        original = match
                        break

            if original is None and len(matches) == 1:
                original = matches[0]

            if original is not None:
                orig_evidence = (
                    original.get("evidence") or {}
                )

                corrected_evidence = dict(
                    corrected.get("evidence") or {}
                )

                corrected_evidence["table_id"] = (
                    orig_evidence.get("table_id")
                )
                corrected_evidence["row_index"] = (
                    orig_evidence.get("row_index")
                )
                corrected_evidence["page"] = (
                    orig_evidence.get("page")
                )

                # Logical 来源字段同样强制恢复，
                # 不接受模型自行漂移。
                for field in (
                    "logical_row_id",
                    "source_row_indices",
                    "source_cell_ids",
                ):
                    if field in orig_evidence:
                        corrected_evidence[field] = (
                            orig_evidence[field]
                        )

                if not corrected_evidence.get("source_text"):
                    corrected_evidence["source_text"] = (
                        orig_evidence.get(
                            "source_text", ""
                        )
                    )

                corrected["evidence"] = corrected_evidence

            restored.append(corrected)

        return restored

    def retry_parameters(
        self,
        failed_parameters: list[dict],
        tables: list[dict],
        logical_input=None,
    ) -> dict:
        """针对校验失败的参数进行定向纠错。

        只把失败参数及其原始来源行发送给 LLM，
        并按有限批量分组、支持截断自动二分；
        纠错失败不阻断主流程，交由现有审核机制。

        Logical 模式下纠错结果仍会重新执行程序化来源映射，
        不能凭模型输出绕过原始来源核对。
        """

        if not failed_parameters:
            return {
                "component_model": "",
                "manufacturer": "",
                "parameters": [],
            }

        tables = self._prepare_tables(
            tables, logical_input=logical_input
        )

        items, unresolvable = self._build_correction_items(
            failed_parameters,
            tables,
        )

        if unresolvable:
            logger.warning(
                "无法自动定位来源的失败参数 %s 个，"
                "交由现有审核机制处理",
                len(unresolvable),
            )

            for param in unresolvable:
                evidence = param.get("evidence") or {}
                logger.warning(
                    "跳过无法定位参数: name=%s, "
                    "table_id=%s, row_index=%s, page=%s",
                    param.get("name"),
                    evidence.get("table_id"),
                    evidence.get("row_index"),
                    evidence.get("page"),
                )

        logger.info(
            "参数纠错: 失败参数=%s, 可定位=%s, 无法定位=%s",
            len(failed_parameters),
            len(items),
            len(unresolvable),
        )

        if not items:
            return {
                "component_model": "",
                "manufacturer": "",
                "parameters": [],
            }

        merged = {
            "component_model": "",
            "manufacturer": "",
            "parameters": [],
        }

        total_batches = (
            len(items) + self.rows_per_chunk - 1
        ) // self.rows_per_chunk

        failed_batches = 0

        for batch_index, start in enumerate(
            range(
                0, len(items), self.rows_per_chunk
            ),
            start=1,
        ):
            batch = items[
                start:start + self.rows_per_chunk
            ]

            try:
                result = self._correct_items(batch)
            except Exception as exc:
                failed_batches += 1
                logger.error(
                    "纠错批次失败: 第 %s/%s 批, "
                    "异常类型=%s: %s",
                    batch_index,
                    total_batches,
                    type(exc).__name__,
                    exc,
                )
                continue

            merged = self._merge_results([
                merged,
                result,
            ])

            logger.info(
                "纠错批次完成: 第 %s/%s 批, "
                "本批参数=%s, 累计参数=%s",
                batch_index,
                total_batches,
                len(result.get("parameters", [])),
                len(merged["parameters"]),
            )

        corrected = self._restore_evidence(
            merged["parameters"],
            failed_parameters,
        )

        # Retry 结果也必须重新执行程序化来源核对。
        corrected = self._canonicalize_source_fields(
            corrected, tables
        )

        logger.info(
            "参数纠错结束: 返回纠错参数=%s, "
            "失败批次=%s",
            len(corrected),
            failed_batches,
        )

        return {
            "component_model": merged["component_model"],
            "manufacturer": merged["manufacturer"],
            "parameters": corrected,
        }

    def get_usage(self) -> dict:
        """返回模型调用 Token 统计。"""

        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": (
                self.prompt_tokens
                + self.completion_tokens
            ),
        }
