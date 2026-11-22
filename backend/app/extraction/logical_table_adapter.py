"""阶段 B3.1：Recovered logical_rows -> LLM 输入适配器。

只做输入适配，不调用 LLM，不修改旧 Evidence 字段语义。

关键点：
- logical_row_id 是逻辑行标识，绝不是 row_index；
- row_indices 使用该逻辑行的首个原始 source_row_index；
- source_row_indices / source_cell_ids 完整保留；
- needs_review=true 的逻辑行默认不进入可信输入；
- 不根据参数名称猜测单位；
- 只包含显式请求的 table_id，避免混入应用示例参数。
"""

import copy
from pathlib import Path


SCHEMA = "logical_table_input/v1"
VERSION = 1

SOURCE_LOGICAL_SCHEMA = "table_logical_rows/v1"
SOURCE_LOGICAL_VERSION = 1

LOGICAL_INPUT_COLUMNS = [
    "logical_row_id",
    "table_id",
    "page",
    "parameter_symbol",
    "parameter_name",
    "test_conditions",
    "min",
    "typ",
    "max",
    "unit",
    "source_row_indices",
    "source_cell_ids",
    "source_text",
    "needs_review",
]


class LogicalTableAdapterError(ValueError):
    """logical_rows 缺失、版本不兼容或无法适配。"""


def _iter_table_groups(logical_rows):
    tables = logical_rows.get("tables")

    if isinstance(tables, list) and tables:
        return tables

    return [logical_rows]


def _validate_source(logical_rows):
    if not isinstance(logical_rows, dict):
        raise LogicalTableAdapterError(
            "logical_rows 必须是 JSON 对象"
        )

    schema = logical_rows.get("schema")

    if schema is not None and schema != SOURCE_LOGICAL_SCHEMA:
        raise LogicalTableAdapterError(
            f"不支持的 logical_rows schema: {schema!r}"
        )

    version = logical_rows.get("logical_rows_version")

    if (
        version is not None
        and version != SOURCE_LOGICAL_VERSION
    ):
        raise LogicalTableAdapterError(
            "不支持的 logical_rows_version: "
            f"{version!r}"
        )

    groups = _iter_table_groups(logical_rows)

    for group in groups:
        if not isinstance(group, dict):
            raise LogicalTableAdapterError(
                "logical_rows 表组必须是 JSON 对象"
            )

        if not isinstance(group.get("logical_rows"), list):
            raise LogicalTableAdapterError(
                "logical_rows 表组缺少 logical_rows 数组"
            )

    return groups


def _exclusion_reason(row, include_needs_review):
    if row.get("needs_review") and not include_needs_review:
        return "needs_review=true，不得作为可信参数"

    if not row.get("logical_row_id"):
        return "缺少 logical_row_id"

    if not row.get("source_row_indices"):
        return "缺少 source_row_indices，无法映射原始行"

    if not row.get("source_cell_ids"):
        return "缺少 source_cell_ids，无法追溯 PDF 来源"

    if not (
        str(row.get("parameter_name") or "").strip()
        or str(row.get("parameter_symbol") or "").strip()
    ):
        return "缺少参数名称/符号"

    values = (
        row.get("min"),
        row.get("typ"),
        row.get("max"),
    )

    if not any(str(value).strip() for value in values):
        return "缺少 min/typ/max 数值"

    return None


def primary_row_index(row):
    """逻辑行的首个原始 source_row_index（无 raw_structure 时的兜底）。"""

    indices = row.get("source_row_indices") or []

    for index in indices:
        if isinstance(index, int):
            return index

    return None


def _resolved_primary_row_index(
    row,
    raw_tables,
    raw_cells,
):
    """优先使用人工修复版本显式给出的 primary_row_index。

    只有该值确实是 source_row_indices 之一时才接受；
    否则回退到基于 PDF cell 几何的选择逻辑。
    """

    indices = row.get("source_row_indices") or []
    explicit = row.get("primary_row_index")

    if (
        isinstance(explicit, int)
        and not isinstance(explicit, bool)
        and explicit in indices
    ):
        return explicit

    return _specific_source_row_index(
        row, raw_tables, raw_cells
    )


def _build_raw_index(raw_structure):
    tables = {}
    cells = {}

    if not isinstance(raw_structure, dict):
        return tables, cells

    for table in raw_structure.get("tables", []):
        table_id = table.get("table_id")
        tables[table_id] = table

        for cell in table.get("cells", []):
            cells[(table_id, cell.get("cell_id"))] = cell

    return tables, cells


def _specific_source_row_index(
    row,
    raw_tables,
    raw_cells,
):
    """用 raw_structure 的 cell 几何选择最具体的 source_row_index。

    选择与 physical_row_band 真正重叠、且 cell 高度最小
    （最具体）的单元格来源行；避免总是落到参数名/单位
    这类跨行 cell 的组起始行。
    """

    indices = [
        index for index in (row.get("source_row_indices") or [])
        if isinstance(index, int)
    ]

    if not indices:
        return None

    band = row.get("physical_row_band") or {}
    top = band.get("top")
    bottom = band.get("bottom")

    if top is None or bottom is None:
        return indices[0]

    table_id = row.get("table_id")
    table = raw_tables.get(table_id)

    if table is None:
        return indices[0]

    header_row_count = int(
        table.get("header_row_count", 0)
    )

    best = None

    for cell_id in row.get("source_cell_ids") or []:
        cell = raw_cells.get((table_id, cell_id))

        if not cell:
            continue

        bbox = cell.get("bbox")

        if (
            not bbox
            or bbox.get("coord_origin") != "TOPLEFT"
        ):
            continue

        cell_top = float(bbox["t"])
        cell_bottom = float(bbox["b"])
        overlap = min(cell_bottom, bottom) - max(
            cell_top, top
        )

        if overlap <= 2.0:
            continue

        row_index = (
            int(cell.get("start_row_offset_idx", 0))
            - header_row_count
        )

        if row_index not in indices:
            continue

        candidate = (
            cell_bottom - cell_top,
            -overlap,
            row_index,
        )

        if best is None or candidate < best:
            best = candidate

    return best[2] if best is not None else indices[0]


def build_source_text(row):
    """由逻辑行已恢复字段构造可追溯的 source_text。

    不改写/猜测任何字段；空字段直接跳过。
    """

    parts = [
        row.get("parameter_symbol") or "",
        row.get("parameter_name") or "",
        row.get("test_conditions") or "",
        row.get("min") or "",
        row.get("typ") or "",
        row.get("max") or "",
        row.get("unit") or "",
    ]

    return " | ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
    )


def _build_record(
    row,
    table_id,
    page,
    include_word_bboxes,
    primary=None,
):
    unit = str(row.get("unit") or "").strip()
    conditions = str(
        row.get("test_conditions") or ""
    ).strip()

    record = {
        "logical_row_id": row.get("logical_row_id"),
        "table_id": table_id,
        "page": page,
        "parameter_symbol": row.get(
            "parameter_symbol", ""
        ),
        "parameter_name": row.get(
            "parameter_name", ""
        ),
        "test_conditions": conditions,
        "min": row.get("min", ""),
        "typ": row.get("typ", ""),
        "max": row.get("max", ""),
        "unit": unit,
        "source_row_indices": list(
            row.get("source_row_indices") or []
        ),
        "source_cell_ids": list(
            row.get("source_cell_ids") or []
        ),
        "source_text": build_source_text(row),
        "needs_review": bool(row.get("needs_review")),
        "primary_row_index": (
            primary
            if primary is not None
            else primary_row_index(row)
        ),
        "unit_missing": not bool(unit),
        "test_conditions_missing": not bool(conditions),
    }

    # 人工修复版本的可选来源细分信息。
    # 不进入 LLM 表格列，只保留在 records/source_map 中。
    if row.get("source_cell_fragments"):
        record["source_cell_fragments"] = copy.deepcopy(
            row.get("source_cell_fragments")
        )

    if row.get("unit_source"):
        record["unit_source"] = copy.deepcopy(
            row.get("unit_source")
        )

    if include_word_bboxes:
        record["source_word_bboxes"] = copy.deepcopy(
            row.get("source_word_bboxes") or []
        )

    return record


def _row_values(record, columns):
    return [record.get(column) for column in columns]


def adapt_logical_rows(
    logical_rows,
    *,
    raw_structure=None,
    table_ids=None,
    include_needs_review=False,
    include_word_bboxes=False,
):
    """把 logical_rows 转为 LLM 可理解的结构化输入。

    参数：
    - logical_rows: StructureRecovery 产出的 logical_rows 文档
    - raw_structure: 可选；用于按 cell 几何选择更精确的
      primary_row_index（推荐传入）
    - table_ids: 只包含指定的 table_id（推荐显式指定通用规格表）
    - include_needs_review: 是否把 needs_review 行也放入输入
      （默认 False；即使放入也会保留 needs_review 标记）
    - include_word_bboxes: 是否附带 source_word_bboxes
    """

    groups = _validate_source(logical_rows)
    raw_tables, raw_cells = _build_raw_index(
        raw_structure
    )

    requested_ids = None

    if table_ids is not None:
        requested_ids = {int(value) for value in table_ids}

    output_tables = []
    total_rows = 0
    total_excluded = 0

    found_ids = set()

    for group in groups:
        table_id = group.get("table_id")
        page = group.get("page")
        page_numbers = group.get("page_numbers")

        if page_numbers is None and page is not None:
            page_numbers = [page]

        if requested_ids is not None and table_id not in requested_ids:
            continue

        found_ids.add(table_id)

        records = []
        excluded = []

        for row in group.get("logical_rows", []):
            reason = _exclusion_reason(
                row, include_needs_review
            )

            if reason is not None:
                excluded.append({
                    "logical_row_id": row.get(
                        "logical_row_id"
                    ),
                    "table_id": table_id,
                    "page": page,
                    "reason": reason,
                    "needs_review": bool(
                        row.get("needs_review")
                    ),
                    "source_row_indices": list(
                        row.get("source_row_indices") or []
                    ),
                    "source_cell_ids": list(
                        row.get("source_cell_ids") or []
                    ),
                })
                continue

            records.append(
                _build_record(
                    row,
                    table_id,
                    page,
                    include_word_bboxes,
                    primary=_resolved_primary_row_index(
                        row, raw_tables, raw_cells
                    ),
                )
            )

        rows = [
            _row_values(record, LOGICAL_INPUT_COLUMNS)
            for record in records
        ]

        row_indices = [
            record["primary_row_index"]
            for record in records
        ]

        source_map = []

        for record in records:
            item = {
                "logical_row_id": record[
                    "logical_row_id"
                ],
                "table_id": record["table_id"],
                "page": record["page"],
                "primary_row_index": record[
                    "primary_row_index"
                ],
                "source_row_indices": record[
                    "source_row_indices"
                ],
                "source_cell_ids": record[
                    "source_cell_ids"
                ],
                "needs_review": record["needs_review"],
            }

            if record.get("source_cell_fragments"):
                item["source_cell_fragments"] = (
                    record["source_cell_fragments"]
                )

            if record.get("unit_source"):
                item["unit_source"] = record[
                    "unit_source"
                ]

            source_map.append(item)

        output_tables.append({
            "table_id": table_id,
            "page": page,
            "page_numbers": list(page_numbers or []),
            "input_mode": "logical",
            "columns": list(LOGICAL_INPUT_COLUMNS),
            "rows": rows,
            "records": records,
            "row_indices": row_indices,
            "row_count": len(records),
            "column_count": len(LOGICAL_INPUT_COLUMNS),
            "source_map": source_map,
            "excluded_rows": excluded,
            "source_logical_rows_schema": group.get(
                "schema", logical_rows.get("schema")
            ),
            "source_logical_rows_version": group.get(
                "logical_rows_version",
                logical_rows.get("logical_rows_version"),
            ),
        })

        total_rows += len(records)
        total_excluded += len(excluded)

    if requested_ids is not None:
        missing = requested_ids - found_ids

        if missing:
            raise LogicalTableAdapterError(
                "请求的 table_id 不在 logical_rows 中: "
                f"{sorted(missing)}"
            )

    return {
        "schema": SCHEMA,
        "logical_table_input_version": VERSION,
        "mode": "logical_rows",
        "tables": output_tables,
        "table_count": len(output_tables),
        "row_count": total_rows,
        "excluded_count": total_excluded,
        "notes": [
            "logical_row_id 是逻辑行标识，不是 row_index，"
            "不得写进 evidence.row_index。",
            "row_indices 使用该逻辑行的最具体原始 "
            "source_row_index（基于 raw_structure 的 cell "
            "几何；未提供时使用首个）；完整来源见各行 "
            "source_row_indices。",
            "needs_review=true 的行默认不进入可信输入。",
            "unit 为空时不得根据参数名称猜测单位。",
            "只包含请求的 table_id，避免把应用示例参数"
            "混作芯片通用规格。",
        ],
    }


def adapt_from_file(
    path,
    *,
    raw_structure=None,
    table_ids=None,
    include_needs_review=False,
    include_word_bboxes=False,
):
    import json

    logical_rows = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    return adapt_logical_rows(
        logical_rows,
        raw_structure=raw_structure,
        table_ids=table_ids,
        include_needs_review=include_needs_review,
        include_word_bboxes=include_word_bboxes,
    )
