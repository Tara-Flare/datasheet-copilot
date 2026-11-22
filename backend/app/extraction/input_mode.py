"""阶段 B3.2a：提取输入模式与 Logical/Flat 表格装配。

设计要点：
- EXTRACTION_INPUT_MODE 只允许 flat / logical，默认 flat；
- flat 模式完全保持旧行为；
- logical 模式只替换 logical_rows 已覆盖的 table_id；
- 未恢复的表格继续使用原始 Flat 表格，绝不丢弃；
- 不在这里调用 LLM，也不修改输入对象。
"""

import logging
import os

from app.extraction.logical_table_adapter import (
    LogicalTableAdapterError,
    adapt_logical_rows,
)


logger = logging.getLogger(__name__)


DEFAULT_MODE = "flat"
VALID_MODES = {"flat", "logical"}


class ExtractionInputModeError(ValueError):
    """提取输入模式或 logical 输入配置不合法。"""


def resolve_extraction_input_mode(value=None):
    """解析 EXTRACTION_INPUT_MODE；只允许显式 flat / logical。"""

    mode = (
        value
        or os.getenv("EXTRACTION_INPUT_MODE", DEFAULT_MODE)
    )
    mode = str(mode).strip().lower()

    if mode not in VALID_MODES:
        raise ExtractionInputModeError(
            "不支持的 EXTRACTION_INPUT_MODE: "
            f"{mode!r}（只允许 flat 或 logical）"
        )

    return mode


def is_logical_table(table):
    """判断表格是否来自 LogicalTableAdapter。"""

    if not isinstance(table, dict):
        return False

    return bool(
        table.get("input_mode") == "logical"
        or table.get("mode") == "logical_rows"
        or isinstance(table.get("records"), list)
    )


def _logical_groups(logical_rows):
    if not isinstance(logical_rows, dict):
        return []

    tables = logical_rows.get("tables")

    if isinstance(tables, list) and tables:
        return tables

    return [logical_rows]


def logical_table_ids(logical_rows):
    """返回 logical_rows 文档中的 table_id 集合。"""

    ids = set()

    for group in _logical_groups(logical_rows):
        table_id = group.get("table_id")

        if isinstance(table_id, int):
            ids.add(table_id)

    return ids


def merge_logical_tables(tables, logical_input):
    """把 LogicalTableAdapter 输出合并到原始 Flat 表格列表。

    只替换输入中已经存在、且被 logical_input 覆盖的 table_id；
    logical_input 中额外出现的表格不会混入本次提取。
    """

    if not isinstance(logical_input, dict):
        raise ExtractionInputModeError(
            "logical_input 必须是 LogicalTableAdapter 输出对象"
        )

    logical_tables = logical_input.get("tables")

    if not isinstance(logical_tables, list):
        raise ExtractionInputModeError(
            "logical_input 缺少 tables 数组"
        )

    by_id = {
        table.get("table_id"): table
        for table in logical_tables
        if isinstance(table, dict)
    }

    merged = []
    replaced_ids = []
    flat_ids = []

    for table in tables:
        table_id = table.get("table_id")
        logical_table = by_id.get(table_id)

        if logical_table is None:
            merged.append(table)
            flat_ids.append(table_id)
        else:
            merged.append(logical_table)
            replaced_ids.append(table_id)

    return {
        "mode": "logical",
        "effective_mode": (
            "logical" if replaced_ids else "flat_fallback"
        ),
        "tables": merged,
        "logical_table_ids": replaced_ids,
        "flat_table_ids": flat_ids,
    }


def build_extraction_input(
    tables,
    *,
    mode=None,
    logical_rows=None,
    raw_structure=None,
    table_ids=None,
    include_needs_review=False,
):
    """按输入模式装配实际发送给 LLM 的表格。

    返回:
    - mode: 请求的模式
    - effective_mode: flat / logical / flat_fallback
    - tables: 实际输入表格
    - logical_table_ids: 被 Logical 输入替换的 table_id
    - flat_table_ids: 继续使用原始 Flat 数据的 table_id
    """

    resolved = resolve_extraction_input_mode(mode)

    if resolved == "flat":
        return {
            "mode": "flat",
            "effective_mode": "flat",
            "tables": list(tables),
            "logical_table_ids": [],
            "flat_table_ids": [
                table.get("table_id") for table in tables
            ],
        }

    if not isinstance(logical_rows, dict):
        raise ExtractionInputModeError(
            "logical 模式缺少 logical_rows"
        )

    if not tables:
        return {
            "mode": "logical",
            "effective_mode": "flat_fallback",
            "tables": [],
            "logical_table_ids": [],
            "flat_table_ids": [],
        }

    input_ids = [
        table.get("table_id")
        for table in tables
        if isinstance(table, dict)
        and isinstance(table.get("table_id"), int)
    ]

    available_ids = logical_table_ids(logical_rows)

    if table_ids is not None:
        requested_ids = {
            int(value) for value in table_ids
        }
        missing = requested_ids - available_ids

        if missing:
            raise LogicalTableAdapterError(
                "请求的 table_id 不在 logical_rows 中: "
                f"{sorted(missing)}"
            )
    else:
        requested_ids = set(input_ids)

    requested_ids &= available_ids
    requested_ids &= set(input_ids)

    if not requested_ids:
        logger.warning(
            "logical 模式没有可用的恢复表格，"
            "全部继续使用原始 Flat 数据"
        )

        return {
            "mode": "logical",
            "effective_mode": "flat_fallback",
            "tables": list(tables),
            "logical_table_ids": [],
            "flat_table_ids": input_ids,
        }

    adapted = adapt_logical_rows(
        logical_rows,
        raw_structure=raw_structure,
        table_ids=sorted(requested_ids),
        include_needs_review=include_needs_review,
    )

    merged = merge_logical_tables(tables, adapted)

    logger.info(
        "logical 输入装配: logical_table_ids=%s, "
        "flat_table_ids=%s",
        merged["logical_table_ids"],
        merged["flat_table_ids"],
    )

    return merged


def build_extraction_tables(tables, **kwargs):
    """便捷入口：只返回实际输入表格。"""

    return build_extraction_input(tables, **kwargs)["tables"]
