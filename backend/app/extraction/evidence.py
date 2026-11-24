
from copy import deepcopy
from collections import Counter
from decimal import Decimal, InvalidOperation
import re


VALUE_COLUMNS = {
    "min": {"min", "minimum"},
    "typ": {"typ", "typical"},
    "max": {"max", "maximum"},
}


def normalize_header(value: str) -> str:
    """标准化表头名称。

    去掉脚注括号，例如 "MIN (Note 9)" 归一化为 "min"，
    否则无法匹配 MIN/TYP/MAX 列。
    """
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)

    return re.sub(
        r"[^a-z]",
        "",
        text
    )


def normalize_unit(unit: str) -> str:
    """统一单位的文字表示。"""
    return (
        str(unit)
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
    )


def canonical_unit_cell(unit: str) -> str:
    """合并单元格中的重复单位。

    Docling 的扁平表格会把同一参数块的多个子行
    合并成一个单位单元格，如 "µA µA µA"。只有所有
    单位 token 完全一致时才折叠为单一单位；不一致
    时保留原文，交给校验器判为待复核。
    """

    text = normalize_unit(unit)
    tokens = text.split()

    if tokens and len(set(tokens)) == 1:
        return tokens[0]

    return text


_PLACEHOLDER_CELL = re.compile(
    r"^(?:l|i|image|<!--\s*image\s*-->)+$",
    re.IGNORECASE,
)
_QUOTE_TOKEN_RE = re.compile(
    r"[a-z0-9.]+|<=|>=|<|>|=|%|\+|-"
)


def _quote_tokens(value: str) -> list[str]:
    text = normalize_text(value).lower()
    text = (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )

    return _QUOTE_TOKEN_RE.findall(text)


def _significant_row_cells(row) -> list[str]:
    """过滤 Docling 占位符与图片标记，保留真实单元格文本。"""

    cells = []

    for cell in row:
        if cell is None:
            continue

        text = re.sub(
            r"<!--\s*image\s*-->",
            " ",
            str(cell),
            flags=re.IGNORECASE,
        ).strip()

        if not text:
            continue

        compact = re.sub(r"\s+", "", text)

        if _PLACEHOLDER_CELL.fullmatch(compact):
            continue

        cells.append(text)

    return cells


def _quote_matches_row(quote: str, row) -> bool:
    """引用与来源行的 token 互相包含即视为一致。

    保留原有“引用可以是行文本子集”的语义；同时允许
    模型在合并行引用中附带 TYP/MAX 等列标签（引用是
    行文本的超集）。两种情况都要求 token 完全对应，
    不接受只命中部分内容的引用。
    """

    if not quote:
        return False

    row_tokens = Counter(
        _quote_tokens(" ".join(_significant_row_cells(row)))
    )
    quote_tokens = Counter(_quote_tokens(quote))

    if not row_tokens or not quote_tokens:
        return False

    quote_inside_row = all(
        count <= row_tokens[token]
        for token, count in quote_tokens.items()
    )
    row_inside_quote = all(
        count <= quote_tokens[token]
        for token, count in row_tokens.items()
    )

    return quote_inside_row or row_inside_quote


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


def _merged_unit_fallback(
    row_index,
    rows,
    unit_column,
    extracted_unit,
):
    """合并单位单元格的兜底解析。

    有些表格只在同一参数块的首行填单位，后续行的
    UNIT 单元格为空。只在同一参数名/符号的相邻行
    中找到相同单位时才接受，避免跨参数猜测。
    """

    if not extracted_unit:
        return ""

    row = rows[row_index]

    def cell_text(candidate, index):
        if index >= len(candidate):
            return ""
        return str(candidate[index] or "").strip().lower()

    symbol = cell_text(row, 0)
    name = cell_text(row, 1)

    for offset in range(1, 5):
        for candidate_index in (
            row_index - offset,
            row_index + offset,
        ):
            if not 0 <= candidate_index < len(rows):
                continue

            candidate = rows[candidate_index]

            if unit_column >= len(candidate):
                continue

            unit = canonical_unit_cell(
                candidate[unit_column]
            )

            if unit != extracted_unit:
                continue

            if symbol and cell_text(
                candidate, 0
            ) != symbol:
                continue

            if name and cell_text(
                candidate, 1
            ) != name:
                continue

            return unit

    return ""


def _identity_qualifier_issue(param_name, row, columns):
    """检查名称与来源行的 shutdown/standby 限定是否一致。

    Flat 路径没有 Logical 身份表，但来源行的参数列本身
    带有 "in Shutdown" / "Standby" 等语义限定。预测名
    缺少或凭空增加该限定时，属于可证明的身份错误，
    不能因为数值和单位相同而放行。
    """

    parameter_cells = []

    for index, column in enumerate(columns):
        normalized = normalize_header(column)

        if "parameter" not in normalized and "symbol" not in normalized:
            continue

        if index < len(row) and row[index]:
            parameter_cells.append(str(row[index]))

    if not parameter_cells:
        return None

    source_text = " ".join(parameter_cells).lower()
    prediction = re.sub(
        r"[^a-z]+", "", str(param_name or "").lower()
    )

    source_limited = (
        "shutdown" in source_text or "standby" in source_text
    )
    prediction_limited = (
        "shutdown" in prediction or "standby" in prediction
    )

    if source_limited == prediction_limited:
        return None

    return (
        "参数身份与来源行限定不一致：来源行"
        + ("包含" if source_limited else "不包含")
        + " shutdown/standby 限定"
    )


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

    identity_issue = _identity_qualifier_issue(
        param.get("name"),
        row,
        columns,
    )

    if identity_issue:
        issues.append(identity_issue)

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

    extracted_unit = normalize_unit(
        param.get("unit", "")
    )
    source_unit = canonical_unit_cell(
        row[unit_column]
    )
    if not source_unit:
        fallback = _merged_unit_fallback(
            row_index,
            rows,
            unit_column,
            extracted_unit,
        )

        if fallback:
            source_unit = fallback

    if source_unit != extracted_unit:
        issues.append(
            f"提取单位 {extracted_unit} "
            f"与来源单位 {source_unit} 不一致"
        )

    # 5. 来源原文检查
    quote = evidence.get(
        "source_text", ""
    ).strip()

    if not _quote_matches_row(quote, row):
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
