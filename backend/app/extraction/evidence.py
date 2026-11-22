
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re


VALUE_COLUMNS = {
    "min": {"min", "minimum"},
    "typ": {"typ", "typical"},
    "max": {"max", "maximum"},
}


def normalize_header(value: str) -> str:
    """标准化表头名称。"""
    return re.sub(
        r"[^a-z]",
        "",
        str(value).lower()
    )


def normalize_unit(unit: str) -> str:
    """统一单位的文字表示。"""
    return (
        str(unit)
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
    )


def normalize_text(value: str) -> str:
    """归一化原文文本，仅处理有明确依据的格式差异。

    LLM 输出的 source_text 常使用 "|" 分隔表格单元格，
    而原始 row_text 用空格拼接单元格，二者语义等价。
    这里仅统一：
    1. 表格 "|" 分隔符 → 空格；
    2. 换行、制表符等空白 → 空格；
    3. 折叠连续空格。
    """
    return re.sub(
        r"\s+",
        " ",
        str(value).replace("|", " "),
    ).strip()


def parse_number(value):
    """
    将单元格内容解析为数字。

    只接受单一数字；
    范围值和复杂表达式交给人工复核。
    """
    text = str(value).strip()

    pattern = (
        r"[+-]?"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
    )

    if not re.fullmatch(pattern, text):
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def find_column(columns, aliases):
    """
    在表头中寻找指定列。

    只返回唯一匹配结果。
    """
    matches = []

    for index, column in enumerate(columns):
        header = normalize_header(column)

        if header in aliases:
            matches.append(index)

    if len(matches) == 1:
        return matches[0]

    return None


def verify_single_parameter(
    param: dict,
    table_map: dict
) -> tuple[str, list[str]]:
    """
    校验单个参数的来源、数值、
    参数类型和单位。
    """

    issues = []

    # B3.2a：Logical 来源映射失败的参数不得沿用
    # Flat 证据路径自动通过。
    mapping_status = param.get(
        "source_mapping_status"
    )

    if mapping_status in {
        "mapped_source_conflict",
        "unknown_logical_row_id",
        "ambiguous_row_index",
        "missing_source",
    }:
        return (
            "pending_review",
            [f"逻辑行来源映射失败: {mapping_status}"],
        )

    evidence = param.get("evidence", {})

    table_id = evidence.get("table_id")
    row_index = evidence.get("row_index")
    page = evidence.get("page")

    # 1. 来源表格检查
    table = table_map.get(table_id)

    if table is None:
        return "invalid", ["来源表格不存在"]

    rows = table.get("rows", [])
    columns = table.get("columns", [])

    if (
        not isinstance(row_index, int)
        or not 0 <= row_index < len(rows)
    ):
        return "invalid", ["来源行号无效"]

    if page not in table.get("page_numbers", []):
        return "invalid", ["来源页码不匹配"]

    row = rows[row_index]

    # 2. 参数类型检查
    value_type = param.get("value_type")

    if value_type == "nominal":
        return (
            "pending_review",
            ["标称值尚未配置对应列规则"]
        )

    aliases = VALUE_COLUMNS.get(value_type)

    if aliases is None:
        return "invalid", ["参数值类型不合法"]

    column_index = find_column(
        columns,
        aliases
    )

    if column_index is None:
        return (
            "pending_review",
            ["无法唯一识别参数值对应的表格列"]
        )

    if column_index >= len(row):
        return "invalid", ["来源行缺少对应单元格"]

    # 3. 检查数值是否与来源列一致
    source_value = parse_number(
        row[column_index]
    )

    if source_value is None:
        return (
            "pending_review",
            ["来源单元格不是可直接比较的单一数字"]
        )

    try:
        extracted_value = Decimal(
            str(param["value"])
        )
    except (InvalidOperation, KeyError):
        return "invalid", ["提取数值无效"]

    if extracted_value != source_value:
        issues.append(
            f"提取数值 {extracted_value} "
            f"与来源数值 {source_value} 不一致"
        )

    # 4. 检查单位
    unit_column = find_column(
        columns,
        {"unit", "units"}
    )

    if unit_column is None or unit_column >= len(row):
        return (
            "pending_review",
            issues + ["无法识别单位列"]
        )

    source_unit = normalize_unit(
        row[unit_column]
    )

    extracted_unit = normalize_unit(
        param.get("unit", "")
    )

    if source_unit != extracted_unit:
        issues.append(
            f"提取单位 {extracted_unit} "
            f"与来源单位 {source_unit} 不一致"
        )

    # 5. 来源原文检查
    row_text = " ".join(
        str(cell)
        for cell in row
        if cell is not None
    )

    quote = evidence.get(
        "source_text", ""
    ).strip()

    if not quote or (
        normalize_text(quote)
        not in normalize_text(row_text)
    ):
        issues.append("原文引用不匹配")

    if issues:
        return "invalid", issues

    return "validated", []


def verify_evidence(
    extraction: dict,
    tables: list[dict]
) -> dict:
    """
    批量检查参数的来源证据。
    """

    result = deepcopy(extraction)

    table_map = {
        table["table_id"]: table
        for table in tables
    }

    for param in result.get("parameters", []):
        status, issues = verify_single_parameter(
            param,
            table_map
        )

        param["validation_status"] = status
        param["validation_errors"] = issues

    return result
