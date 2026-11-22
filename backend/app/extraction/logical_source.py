"""阶段 B3.2a：逻辑行提取结果的程序化来源映射。

原则：
- logical_row_id 是逻辑行标识，不是 row_index；
- 优先用 logical_row_id 做唯一映射；
- 没有 logical_row_id 时，只有 source row_index 唯一命中才映射；
- 多条逻辑行共享同一 row_index（如 t9.p30 / t9.p31）时不得猜测；
- LLM 输出的来源字段只作为线索，最终 table_id / row_index /
  page / source_row_indices / source_cell_ids / source_text
  全部由程序按 canonical record 重写；
- 无法核对的来源明确标记，交由校验/审核，不静默信任。
"""

from collections import defaultdict
from decimal import Decimal, InvalidOperation

from app.extraction.input_mode import is_logical_table


MAPPED = "mapped"
MAPPED_CORRECTED = "mapped_corrected"
MAPPED_SOURCE_CONFLICT = "mapped_source_conflict"
UNKNOWN_LOGICAL_ROW_ID = "unknown_logical_row_id"
AMBIGUOUS_ROW_INDEX = "ambiguous_row_index"
MISSING_SOURCE = "missing_source"

UNTRUSTED_STATUSES = {
    MAPPED_SOURCE_CONFLICT,
    UNKNOWN_LOGICAL_ROW_ID,
    AMBIGUOUS_ROW_INDEX,
    MISSING_SOURCE,
}


def _to_decimal(value):
    text = str(value).strip()

    text = (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )

    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, AttributeError):
        return None


def _normalize_unit(unit):
    return (
        str(unit or "")
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
    )


def _index_records(tables):
    by_logical_id = {}
    by_table = defaultdict(lambda: defaultdict(list))

    for table in tables:
        if not is_logical_table(table):
            continue

        table_id = table.get("table_id")
        records = table.get("records") or []

        for position, record in enumerate(records):
            if not isinstance(record, dict):
                continue

            logical_row_id = record.get("logical_row_id")

            if not logical_row_id:
                continue

            info = {
                "table": table,
                "record": record,
                "position": position,
                "table_id": table_id,
            }

            # logical_row_id 必须全局唯一；重复时不自动选择。
            by_logical_id.setdefault(
                logical_row_id, []
            ).append(info)

            by_table[table_id][
                record.get("primary_row_index")
            ].append(info)

    return by_logical_id, by_table


def _mark_untrusted(param, status, details=None):
    param["source_mapping_status"] = status
    param["source_mapping"] = {
        "status": status,
        "trusted": False,
        "details": details or {},
    }

    return param


def _field_check(provided, canonical, *, provided_present=True):
    if not provided_present:
        return None

    return provided == canonical


def _canonicalize_one(
    param,
    by_logical_id,
    by_table,
):
    evidence = param.get("evidence")

    if not isinstance(evidence, dict):
        return param

    table_id = evidence.get("table_id")
    logical_row_id = evidence.get("logical_row_id")
    row_index = evidence.get("row_index")

    candidates = []
    method = None

    if logical_row_id:
        candidates = by_logical_id.get(
            logical_row_id, []
        )
        method = "logical_row_id"

        if not candidates:
            return _mark_untrusted(
                param,
                UNKNOWN_LOGICAL_ROW_ID,
                {
                    "logical_row_id": logical_row_id,
                    "table_id": table_id,
                    "message": (
                        "LLM 提供的 logical_row_id 不在可信 "
                        "Logical 输入中（可能编造或属于 "
                        "needs_review 行）"
                    ),
                },
            )

        if len(candidates) > 1:
            return _mark_untrusted(
                param,
                AMBIGUOUS_ROW_INDEX,
                {
                    "logical_row_id": logical_row_id,
                    "message": (
                        "logical_row_id 在输入中重复，"
                        "无法唯一映射"
                    ),
                },
            )
    else:
        table_candidates = by_table.get(table_id, {})

        if table_candidates:
            candidates = table_candidates.get(
                row_index, []
            )

            if not candidates:
                return _mark_untrusted(
                    param,
                    MISSING_SOURCE,
                    {
                        "table_id": table_id,
                        "row_index": row_index,
                        "message": (
                            "没有 logical_row_id，且 "
                            "row_index 未命中任何逻辑行"
                        ),
                    },
                )

            if len(candidates) > 1:
                return _mark_untrusted(
                    param,
                    AMBIGUOUS_ROW_INDEX,
                    {
                        "table_id": table_id,
                        "row_index": row_index,
                        "candidate_logical_row_ids": [
                            item["record"].get(
                                "logical_row_id"
                            )
                            for item in candidates
                        ],
                        "message": (
                            "多个逻辑行共享同一 row_index，"
                            "必须用 logical_row_id 区分"
                        ),
                    },
                )

            method = "source_row_index"
        else:
            # 该 table_id 不在 Logical 输入中，保持 Flat 行为。
            return param

    canonical = candidates[0]
    table = canonical["table"]
    record = canonical["record"]

    canonical_table_id = table.get("table_id")
    canonical_row_index = record.get(
        "primary_row_index"
    )
    canonical_page = record.get("page")
    canonical_source_row_indices = list(
        record.get("source_row_indices") or []
    )
    canonical_source_cell_ids = list(
        record.get("source_cell_ids") or []
    )
    canonical_source_text = record.get(
        "source_text", ""
    )

    checks = {
        "logical_row_id": _field_check(
            logical_row_id,
            record.get("logical_row_id"),
            provided_present=bool(logical_row_id),
        ),
        "table_id": _field_check(
            table_id,
            canonical_table_id,
            provided_present=table_id is not None,
        ),
        "row_index": _field_check(
            row_index,
            canonical_row_index,
            provided_present=row_index is not None,
        ),
        "page": _field_check(
            evidence.get("page"),
            canonical_page,
            provided_present=evidence.get("page") is not None,
        ),
        "source_row_indices": _field_check(
            evidence.get("source_row_indices"),
            canonical_source_row_indices,
            provided_present=(
                evidence.get("source_row_indices") is not None
            ),
        ),
        "source_cell_ids": _field_check(
            evidence.get("source_cell_ids"),
            canonical_source_cell_ids,
            provided_present=(
                evidence.get("source_cell_ids") is not None
            ),
        ),
    }

    value_type = param.get("value_type")
    declared_value = None

    if value_type in ("min", "typ", "max"):
        declared_value = _to_decimal(
            record.get(value_type)
        )

    extracted_value = _to_decimal(param.get("value"))

    checks["value_column"] = (
        declared_value is not None
        and extracted_value is not None
        and declared_value == extracted_value
    )

    checks["unit"] = (
        _normalize_unit(param.get("unit"))
        == _normalize_unit(record.get("unit"))
    )

    source_fields = (
        "table_id",
        "page",
        "row_index",
        "source_row_indices",
        "source_cell_ids",
    )
    source_conflict = any(
        checks.get(field) is False
        for field in source_fields
    )
    content_mismatch = any(
        checks.get(field) is False
        for field in ("value_column", "unit")
    )

    if source_conflict:
        mapping_status = MAPPED_SOURCE_CONFLICT
    elif content_mismatch:
        mapping_status = MAPPED_CORRECTED
    else:
        mapping_status = MAPPED

    new_evidence = dict(evidence)
    new_evidence["table_id"] = canonical_table_id
    new_evidence["row_index"] = canonical_row_index
    new_evidence["page"] = canonical_page
    new_evidence["source_text"] = canonical_source_text
    new_evidence["logical_row_id"] = record.get(
        "logical_row_id"
    )
    new_evidence["source_row_indices"] = (
        canonical_source_row_indices
    )
    new_evidence["source_cell_ids"] = (
        canonical_source_cell_ids
    )

    param["evidence"] = new_evidence
    param["source_mapping_status"] = mapping_status
    param["source_mapping"] = {
        "status": mapping_status,
        "trusted": not source_conflict,
        "method": method,
        "logical_row_id": record.get(
            "logical_row_id"
        ),
        "table_id": canonical_table_id,
        "row_index": canonical_row_index,
        "preserved_source_row_indices": (
            canonical_source_row_indices
        ),
        "preserved_source_cell_ids": (
            canonical_source_cell_ids
        ),
        "provided_source": {
            "table_id": table_id,
            "page": evidence.get("page"),
            "row_index": row_index,
            "source_row_indices": evidence.get(
                "source_row_indices"
            ),
            "source_cell_ids": evidence.get(
                "source_cell_ids"
            ),
        },
        "checks": checks,
    }

    return param


def canonicalize_parameters(parameters, tables):
    """按 Logical 输入重写参数来源字段。

    只处理 Logical 表格中的参数；Flat 表格参数保持原样。
    """

    logical_tables = [
        table for table in tables
        if is_logical_table(table)
    ]

    if not logical_tables:
        return parameters

    by_logical_id, by_table = _index_records(
        logical_tables
    )

    canonical = []

    for param in parameters:
        if not isinstance(param, dict):
            canonical.append(param)
            continue

        canonical.append(
            _canonicalize_one(
                param, by_logical_id, by_table
            )
        )

    return canonical


def is_trusted_mapping(param):
    """判断参数是否具备可信的逻辑行来源映射。"""

    if not isinstance(param, dict):
        return False

    status = param.get("source_mapping_status")

    if status is None:
        return False

    return status not in UNTRUSTED_STATUSES
