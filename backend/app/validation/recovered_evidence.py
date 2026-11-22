"""阶段 B2.1：原始提取参数与逻辑行的离线证据匹配。

只读评估：不调用 LLM，不重新解析 PDF，不修改原始提取结果。
"""

import copy
import re
from decimal import Decimal, InvalidOperation


SCHEMA = "recovered_evidence_validation/v1"
VERSION = 1

# 原始参数名 -> 参数族
ORIGINAL_FAMILY = {
    "quiescent_current": "ground_current",
    "output_voltage_accuracy": "accuracy",
    "dropout_voltage": "dropout",
    "output_current_limit": "current_limit",
    "psrr": "psrr",
    "input_voltage": "input_voltage",
    "output_voltage": "output_voltage",
    "output_current": "output_current",
    "operating_temperature": "temperature",
    "input_current": "input_current",
    "output_load": "output_load",
    "uvlo threshold": "uvlo_threshold",
    "uvlo_threshold": "uvlo_threshold",
    "uvlo hysteresis": "uvlo_hysteresis",
    "uvlo_hysteresis": "uvlo_hysteresis",
}


def _to_decimal(value):
    text = str(value).strip()
    # Docling/PDF 文本常用 Unicode 负号
    text = (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )

    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, AttributeError):
        return None


def normalize_unit(unit):
    return (
        str(unit or "")
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
    )


def _normalize_text(text):
    value = str(text or "").lower()

    replacements = [
        ("μ", "u"),
        ("µ", "u"),
        ("×", "x"),
        ("–", "-"),
        ("−", "-"),
        ("≥", ">="),
        ("≤", "<="),
        ("°c", "c"),
        ("°", ""),
    ]

    for source, target in replacements:
        value = value.replace(source, target)

    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*([<>=])\s*", r"\1", value)

    return value.strip()


def _tokens(text):
    return set(
        re.findall(
            r"[a-z0-9.]+|<=|>=|<|>|=|%|\+|-",
            text,
        )
    )


_NUMBER_PATTERN = r"[+-]?\d+(?:\.\d+)?"
_UNIT_PATTERN = r"[a-z%]+"
_VARIABLE_PATTERN = r"[a-z][a-z0-9_]*(?:\([a-z0-9_]+\))?"

_RANGE_RE = re.compile(
    rf"^(?P<lower>{_NUMBER_PATTERN})\s*"
    rf"(?P<lower_unit>{_UNIT_PATTERN})?\s*"
    rf"(?P<lower_op><=|>=|<|>)\s*"
    rf"(?P<variable>{_VARIABLE_PATTERN})\s*"
    rf"(?P<upper_op><=|>=|<|>)\s*"
    rf"(?P<upper>{_NUMBER_PATTERN})\s*"
    rf"(?P<upper_unit>{_UNIT_PATTERN})?$"
)

_TO_RANGE_RE = re.compile(
    rf"^(?P<variable>{_VARIABLE_PATTERN})\s*=\s*"
    rf"(?P<lower>{_NUMBER_PATTERN})\s*"
    rf"(?P<lower_unit>{_UNIT_PATTERN})?\s*to\s*"
    rf"(?P<upper>{_NUMBER_PATTERN})\s*"
    rf"(?P<upper_unit>{_UNIT_PATTERN})?$"
)

_EQUAL_RE = re.compile(
    rf"^(?P<variable>{_VARIABLE_PATTERN})\s*=\s*"
    rf"(?P<value>{_NUMBER_PATTERN})\s*"
    rf"(?P<unit>{_UNIT_PATTERN})?$"
)

_COMPARE_RE = re.compile(
    rf"^(?P<variable>{_VARIABLE_PATTERN})\s*"
    rf"(?P<op><=|>=|<|>)\s*"
    rf"(?P<value>{_NUMBER_PATTERN})\s*"
    rf"(?P<unit>{_UNIT_PATTERN})?$"
)

_UNIT_ALIASES = {
    "v": "v",
    "mv": "mv",
    "uv": "uv",
    "kv": "kv",
    "a": "a",
    "ma": "ma",
    "ua": "ua",
    "na": "na",
    "c": "c",
    "hz": "hz",
    "khz": "khz",
    "mhz": "mhz",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "db": "db",
    "%": "%",
}


def _normalize_condition_unit(unit):
    if not unit:
        return ""

    value = (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("°", "")
    )

    return _UNIT_ALIASES.get(value, value)


def _normalize_variable(variable):
    return re.sub(r"\(.*\)", "", str(variable or "").lower())


def _text_variable(text):
    match = re.match(
        r"^([a-z][a-z0-9_]*(?:\([a-z0-9_]+\))?)\s*=",
        text,
    )

    return _normalize_variable(match.group(1)) if match else None


def _parse_constraint(segment):
    text = re.sub(r"\(\d+\)", "", segment)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    match = _RANGE_RE.match(text)

    if match:
        return {
            "kind": "range",
            "variable": _normalize_variable(
                match.group("variable")
            ),
            "lower": _to_decimal(match.group("lower")),
            "lower_op": match.group("lower_op"),
            "lower_unit": _normalize_condition_unit(
                match.group("lower_unit")
            ),
            "upper": _to_decimal(match.group("upper")),
            "upper_op": match.group("upper_op"),
            "upper_unit": _normalize_condition_unit(
                match.group("upper_unit")
            ),
            "text": text,
        }

    match = _TO_RANGE_RE.match(text)

    if match:
        return {
            "kind": "range",
            "variable": _normalize_variable(
                match.group("variable")
            ),
            "lower": _to_decimal(match.group("lower")),
            "lower_op": ">=",
            "lower_unit": _normalize_condition_unit(
                match.group("lower_unit")
            ),
            "upper": _to_decimal(match.group("upper")),
            "upper_op": "<=",
            "upper_unit": _normalize_condition_unit(
                match.group("upper_unit")
            ),
            "text": text,
        }

    match = _EQUAL_RE.match(text)

    if match:
        return {
            "kind": "equal",
            "variable": _normalize_variable(
                match.group("variable")
            ),
            "value": _to_decimal(match.group("value")),
            "unit": _normalize_condition_unit(
                match.group("unit")
            ),
            "text": text,
        }

    match = _COMPARE_RE.match(text)

    if match:
        return {
            "kind": "cmp",
            "variable": _normalize_variable(
                match.group("variable")
            ),
            "op": match.group("op"),
            "value": _to_decimal(match.group("value")),
            "unit": _normalize_condition_unit(
                match.group("unit")
            ),
            "text": text,
        }

    return {
        "kind": "text",
        "variable": _text_variable(text),
        "text": text,
    }


def parse_conditions(text):
    constraints = []
    seen = set()

    for segment in re.split(
        r"[|;,]", _normalize_text(text)
    ):
        segment = segment.strip()

        if not segment:
            continue

        constraint = _parse_constraint(segment)

        if constraint is None:
            continue

        key = _constraint_key(constraint)

        if key in seen:
            continue

        seen.add(key)
        constraints.append(constraint)

    return constraints


def _constraint_key(constraint):
    if constraint["kind"] == "range":
        return (
            "range",
            constraint["variable"],
            constraint["lower_op"],
            str(constraint["lower"]),
            constraint["lower_unit"],
            constraint["upper_op"],
            str(constraint["upper"]),
            constraint["upper_unit"],
        )

    if constraint["kind"] == "equal":
        return (
            "equal",
            constraint["variable"],
            str(constraint["value"]),
            constraint["unit"],
        )

    if constraint["kind"] == "cmp":
        return (
            "cmp",
            constraint["variable"],
            constraint["op"],
            str(constraint["value"]),
            constraint["unit"],
        )

    return ("text", constraint["text"])


def _text_matches(first, second):
    if first["text"] == second["text"]:
        return True

    return (
        first["text"] in second["text"]
        or second["text"] in first["text"]
    )


def evaluate_conditions(original, logical):
    """结构化比较原始条件与逻辑行条件。

    返回 status:
    - match: 完整条件一致（允许格式/顺序差异）
    - missing: 原始条件缺少逻辑行中的必要约束
    - conflict: 同一变量/单位的约束冲突
    - ambiguous: 无法可靠解析
    """

    original_constraints = parse_conditions(original)
    logical_constraints = parse_conditions(logical)

    original_text = _normalize_text(original)
    logical_text = _normalize_text(logical)

    if not original_text:
        return {
            "status": "missing",
            "missing_conditions": ["原始测试条件为空"],
            "conflicting_conditions": [],
            "original_constraints": [],
            "logical_constraints": [
                item["text"] for item in logical_constraints
            ],
        }

    if not logical_text:
        return {
            "status": "ambiguous",
            "missing_conditions": [],
            "conflicting_conditions": [],
            "reason": "逻辑行测试条件为空",
            "original_constraints": [
                item["text"] for item in original_constraints
            ],
            "logical_constraints": [],
        }

    original_keys = {
        _constraint_key(item)
        for item in original_constraints
    }
    logical_keys = {
        _constraint_key(item)
        for item in logical_constraints
    }

    missing = []
    conflicts = []

    for item in logical_constraints:
        if _constraint_key(item) in original_keys:
            continue

        if item["kind"] == "text" and any(
            other["kind"] == "text"
            and _text_matches(item, other)
            for other in original_constraints
        ):
            continue

        same_variable = [
            other for other in original_constraints
            if item["variable"]
            and other["variable"] == item["variable"]
        ]

        if same_variable:
            conflicts.append(item["text"])
        else:
            missing.append(item["text"])

    for item in original_constraints:
        if _constraint_key(item) in logical_keys:
            continue

        if item["kind"] == "text" and any(
            other["kind"] == "text"
            and _text_matches(item, other)
            for other in logical_constraints
        ):
            continue

        same_variable = [
            other for other in logical_constraints
            if item["variable"]
            and other["variable"] == item["variable"]
        ]

        if same_variable and item["text"] not in conflicts:
            conflicts.append(item["text"])

    if conflicts:
        status = "conflict"
    elif missing:
        status = "missing"
    else:
        status = "match"

    return {
        "status": status,
        "missing_conditions": missing,
        "conflicting_conditions": conflicts,
        "original_constraints": [
            item["text"] for item in original_constraints
        ],
        "logical_constraints": [
            item["text"] for item in logical_constraints
        ],
    }


def condition_supported(original, logical):
    """完整条件是否一致（保留旧接口）。"""

    return evaluate_conditions(
        original, logical
    )["status"] == "match"


def original_family(name):
    key = str(name or "").strip().lower()
    return ORIGINAL_FAMILY.get(key, key)


def logical_family(row):
    symbol = str(row.get("parameter_symbol") or "").upper()
    name = str(row.get("parameter_name") or "").lower()

    if "HYST" in symbol or "hysteresis" in name:
        return "uvlo_hysteresis"

    if "VUVLO" in symbol or "uvlo" in name:
        return "uvlo_threshold"

    if symbol == "IGND/IOUT":
        return "ground_current_load"

    if "IGND" in symbol:
        return "ground_current"

    if symbol == "VDO" or "dropout voltage" in name:
        return "dropout"

    if symbol == "ICL":
        return "current_limit"

    if symbol == "PSRR":
        return "psrr"

    if symbol == "VIN" or "input voltage" in name:
        return "input_voltage"

    if symbol == "VOUT" or "output voltage" in name:
        return "output_voltage"

    if symbol == "IOUT" or "output current" in name:
        return "output_current"

    if symbol == "TJ" or "junction temperature" in name:
        return "temperature"

    if "accuracy" in name:
        return "accuracy"

    return name or symbol or "unknown"


def _value_match(param, row):
    value_type = param.get("value_type")
    value = _to_decimal(param.get("value"))

    columns = ("min", "typ", "max")
    column_values = {
        column: _to_decimal(row.get(column))
        for column in columns
    }

    declared = (
        column_values.get(value_type)
        if value_type in columns
        else None
    )

    if value is not None and declared is not None:
        if declared == value:
            return {
                "status": "match",
                "value_type": value_type,
                "value": str(value),
                "logical_value": row.get(value_type),
            }

        other = [
            column for column in columns
            if column != value_type
            and column_values.get(column) is not None
            and column_values[column] == value
        ]

        if other:
            return {
                "status": "value_type_conflict",
                "value_type": value_type,
                "value": str(value),
                "matching_columns": other,
                "logical_values": {
                    column: row.get(column)
                    for column in columns
                },
            }

        return {
            "status": "conflict",
            "value_type": value_type,
            "value": str(value),
            "logical_value": row.get(value_type),
            "logical_values": {
                column: row.get(column)
                for column in columns
            },
        }

    if value is not None and declared is None:
        other = [
            column for column in columns
            if column_values.get(column) is not None
            and column_values[column] == value
        ]

        if other:
            return {
                "status": "value_type_conflict",
                "value_type": value_type,
                "value": str(value),
                "matching_columns": other,
                "logical_values": {
                    column: row.get(column)
                    for column in columns
                },
            }

        return {
            "status": "unsupported",
            "value_type": value_type,
            "value": str(value),
            "reason": "逻辑行对应列无值",
        }

    return {
        "status": "unsupported",
        "reason": "原始数值无法解析",
    }


def _unit_match(param, row):
    original = normalize_unit(param.get("unit"))
    logical = normalize_unit(row.get("unit"))

    if original and logical:
        if original == logical:
            return {
                "status": "match",
                "unit": original,
            }

        return {
            "status": "conflict",
            "unit": original,
            "logical_unit": logical,
        }

    if not original and logical:
        return {
            "status": "missing",
            "logical_unit": logical,
        }

    if original and not logical:
        return {
            "status": "unsupported",
            "unit": original,
            "reason": "逻辑行单位为空",
        }

    return {
        "status": "unsupported",
        "reason": "双方单位均为空",
    }


def _identity_match(param, row):
    original = original_family(param.get("name"))
    logical = logical_family(row)

    return {
        "status": "match" if original == logical else "conflict",
        "original_family": original,
        "logical_family": logical,
        "logical_symbol": row.get("parameter_symbol"),
        "logical_name": row.get("parameter_name"),
    }


def _conditions_match(param, row):
    original = param.get("test_conditions", "")
    logical = row.get("test_conditions", "")

    result = evaluate_conditions(original, logical)
    result["original"] = original
    result["logical"] = logical

    return result


def _source_match(param, row, raw_cells=None, table_id=None):
    evidence = param.get("evidence") or {}

    checks = {
        "table_id_match": (
            evidence.get("table_id") == row.get("table_id")
        ),
        "page_match": (
            evidence.get("page") == row.get("page")
        ),
        "has_source_cells": bool(row.get("source_cell_ids")),
        "has_source_words": bool(row.get("source_word_bboxes")),
    }

    # B3.2a：若输入携带 Logical 来源字段，必须与
    # canonical logical row 完全一致。
    if evidence.get("logical_row_id") is not None:
        checks["logical_row_id_match"] = (
            evidence.get("logical_row_id")
            == row.get("logical_row_id")
        )

    if evidence.get("source_row_indices") is not None:
        checks["source_row_indices_match"] = (
            list(evidence.get("source_row_indices"))
            == list(row.get("source_row_indices") or [])
        )

    if raw_cells is not None:
        checks["cell_ids_exist"] = all(
            (table_id, cell_id) in raw_cells
            for cell_id in row.get("source_cell_ids", [])
        )

    ok = all(checks.values())

    return {
        "status": "match" if ok else "unsupported",
        "checks": checks,
        "source_cell_ids": row.get("source_cell_ids", []),
        "source_word_bboxes": row.get("source_word_bboxes", []),
    }


def _score_candidate(param, row):
    score = 0.0

    value = _value_match(param, row)

    if value["status"] == "match":
        score += 20
    elif value["status"] == "value_type_conflict":
        score += 5
    elif value["status"] == "conflict":
        score -= 10

    unit = _unit_match(param, row)

    if unit["status"] == "match":
        score += 5
    elif unit["status"] == "conflict":
        score -= 2

    identity = _identity_match(param, row)

    if identity["status"] == "match":
        score += 3

    conditions = _conditions_match(param, row)

    if conditions["status"] == "match":
        score += 4

    if row.get("needs_review"):
        score -= 5

    return score


class RecoveredEvidenceValidator:
    """用 logical_rows 对原始提取参数做离线证据校验。"""

    def __init__(
        self,
        logical_rows,
        raw_structure=None,
        baseline=None,
    ):
        self.logical_rows = logical_rows
        self.raw_structure = raw_structure
        self.baseline = baseline
        self.rows = logical_rows.get("logical_rows", [])

        self.raw_cells = {}

        if raw_structure is not None:
            for table in raw_structure.get("tables", []):
                table_id = table.get("table_id")

                for cell in table.get("cells", []):
                    self.raw_cells[
                        (table_id, cell.get("cell_id"))
                    ] = cell

    def _candidates(self, param):
        evidence = param.get("evidence") or {}
        table_id = evidence.get("table_id")
        row_index = evidence.get("row_index")
        logical_row_id = evidence.get("logical_row_id")

        # B3.2a：logical_row_id 是强来源键。
        # 一旦提供，只按它做精确匹配；不得在 ID 编造时
        # 悄悄回退到 row_index（多条逻辑行可能共享 row_index）。
        if logical_row_id:
            return [
                row for row in self.rows
                if row.get("table_id") == table_id
                and row.get("logical_row_id")
                == logical_row_id
            ]

        candidates = [
            row for row in self.rows
            if row.get("table_id") == table_id
            and row_index in row.get("source_row_indices", [])
        ]

        return candidates

    def _table_covered(self, table_id):
        return any(
            row.get("table_id") == table_id
            for row in self.rows
        )

    def _select(self, param, candidates):
        scored = sorted(
            (
                (_score_candidate(param, row), row)
                for row in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        if not scored:
            return None, [], "no_candidate"

        best_score, best_row = scored[0]

        ties = [
            row for score, row in scored[1:]
            if score == best_score
        ]

        if ties:
            return None, [row for _, row in scored], "ambiguous"

        return best_row, [row for _, row in scored], "unique"

    def _validate_param(self, index, param, baseline_param):
        evidence = param.get("evidence") or {}
        table_id = evidence.get("table_id")

        candidates = self._candidates(param)
        selected, scored_candidates, mapping = self._select(
            param, candidates
        )

        baseline_status = (
            baseline_param.get("validation_status")
            if baseline_param else None
        )

        result = {
            "index": index,
            "name": param.get("name"),
            "value": param.get("value"),
            "unit": param.get("unit"),
            "value_type": param.get("value_type"),
            "test_conditions": param.get("test_conditions"),
            "evidence": evidence,
            "baseline_status": baseline_status,
            "baseline_errors": (
                baseline_param.get("validation_errors", [])
                if baseline_param else []
            ),
            "candidate_logical_row_ids": [
                row.get("logical_row_id")
                for row in scored_candidates
            ],
            "mapping": {
                "method": mapping,
                "candidate_count": len(candidates),
                "matched_by": (
                    "logical_row_id"
                    if evidence.get("logical_row_id")
                    else "row_index"
                ),
            },
            "selected_logical_row_id": (
                selected.get("logical_row_id")
                if selected else None
            ),
            "field_checks": {},
            "conflicts": [],
            "correction_suggestions": [],
            "recovered_status": None,
            "reason": None,
        }

        # B3.2a：Logical 来源映射已明确失败时，
        # 即使数值碰巧能匹配，也不得自动通过。
        mapping_status = param.get(
            "source_mapping_status"
        )

        if mapping_status in {
            "mapped_source_conflict",
            "unknown_logical_row_id",
            "ambiguous_row_index",
            "missing_source",
        }:
            result["recovered_status"] = "pending_review"
            result["reason"] = (
                "Logical 来源映射不可信: "
                f"{mapping_status}"
            )
            return result

        if not self._table_covered(table_id):
            result["recovered_status"] = "not_covered"
            result["reason"] = (
                "logical_rows 未覆盖该 table_id，"
                "无法用恢复证据重新校验"
            )
            return result

        if selected is None:
            result["recovered_status"] = "pending_review"
            result["reason"] = (
                "无法唯一映射到逻辑行"
                if mapping == "ambiguous"
                else "找不到匹配的逻辑行"
            )
            return result

        value = _value_match(param, selected)
        unit = _unit_match(param, selected)
        identity = _identity_match(param, selected)
        conditions = _conditions_match(param, selected)
        source = _source_match(
            param, selected, self.raw_cells, table_id
        )
        needs_review = {
            "status": (
                "conflict"
                if selected.get("needs_review")
                else "match"
            ),
            "logical_row_needs_review": bool(
                selected.get("needs_review")
            ),
        }

        result["field_checks"] = {
            "identity": identity,
            "value": value,
            "unit": unit,
            "value_type": {
                "status": (
                    "match"
                    if value["status"] == "match"
                    else value["status"]
                ),
                "declared": param.get("value_type"),
            },
            "test_conditions": conditions,
            "source": source,
            "needs_review": needs_review,
        }

        if selected.get("needs_review"):
            result["recovered_status"] = "pending_review"
            result["reason"] = "逻辑行 needs_review=true，禁止自动通过"
            return result

        if source["status"] != "match":
            result["recovered_status"] = "pending_review"
            result["reason"] = "逻辑行 PDF 来源不完整或页码/表格不匹配"
            return result

        conflicts = []

        if identity["status"] != "match":
            conflicts.append("identity")

        if value["status"] == "conflict":
            conflicts.append("value")
        elif value["status"] == "value_type_conflict":
            conflicts.append("value_type")

        if unit["status"] == "conflict":
            conflicts.append("unit")

        condition_status = conditions.get("status")

        if condition_status == "conflict":
            conflicts.append("test_conditions")

        missing_fields = []

        if unit["status"] == "missing":
            missing_fields.append("unit")

        if condition_status == "missing":
            missing_fields.append("test_conditions")

        result["conflicts"] = conflicts
        result["missing_fields"] = missing_fields

        # 纠错建议
        if conflicts or missing_fields:
            result["correction_suggestions"].append({
                "logical_row_id": selected.get("logical_row_id"),
                "parameter_symbol": selected.get(
                    "parameter_symbol"
                ),
                "parameter_name": selected.get(
                    "parameter_name"
                ),
                "value_type": param.get("value_type"),
                "value": selected.get(
                    param.get("value_type")
                ),
                "unit": selected.get("unit"),
                "test_conditions": selected.get(
                    "test_conditions"
                ),
                "missing_fields": missing_fields,
                "missing_conditions": conditions.get(
                    "missing_conditions", []
                ),
                "conflicting_conditions": conditions.get(
                    "conflicting_conditions", []
                ),
                "source_cell_ids": selected.get(
                    "source_cell_ids", []
                ),
                "source_word_bboxes": selected.get(
                    "source_word_bboxes", []
                ),
                "fields_to_fix": conflicts + missing_fields,
            })

        if conflicts:
            result["recovered_status"] = "invalid"

            if "test_conditions" in conflicts:
                result["reason"] = (
                    "测试条件与逻辑行存在明确冲突"
                )
            else:
                result["reason"] = (
                    "参数身份/数值/单位与逻辑行冲突"
                )
        elif value["status"] != "match":
            result["recovered_status"] = "pending_review"
            result["reason"] = "数值无法由逻辑行唯一支持"
        elif condition_status == "ambiguous":
            result["recovered_status"] = "pending_review"
            result["reason"] = "测试条件存在歧义"
        elif (
            unit["status"] == "unsupported"
            or condition_status == "unsupported"
        ):
            result["recovered_status"] = "pending_review"
            result["reason"] = "单位或测试条件无法由逻辑行支持"
        elif missing_fields:
            result["recovered_status"] = "correctable"

            if "test_conditions" in missing_fields:
                result["reason"] = (
                    "原始测试条件不完整（缺少必要约束）"
                )
            else:
                result["reason"] = (
                    "原始单位缺失，可由逻辑行补全"
                )
        else:
            result["recovered_status"] = "validated"
            result["reason"] = (
                "身份、数值、单位、条件、来源均完整匹配"
            )

        return result

    def validate(self, extraction, baseline=None):
        params = extraction.get("parameters", [])
        baseline = baseline or self.baseline or {}
        baseline_params = baseline.get("parameters", [])

        results = []

        for index, param in enumerate(params):
            baseline_param = (
                baseline_params[index]
                if index < len(baseline_params)
                else None
            )
            results.append(
                self._validate_param(
                    index, param, baseline_param
                )
            )

        return self._build_report(results)

    def _build_report(self, results):
        total = len(results)

        covered = [
            result for result in results
            if result["recovered_status"] != "not_covered"
        ]
        not_covered = [
            result for result in results
            if result["recovered_status"] == "not_covered"
        ]

        baseline_validated = sum(
            1
            for result in results
            if result["baseline_status"] == "validated"
        )
        baseline_validated_covered = sum(
            1
            for result in covered
            if result["baseline_status"] == "validated"
        )

        recovered_validated = sum(
            1
            for result in covered
            if result["recovered_status"] == "validated"
        )

        newly_validated = sum(
            1
            for result in covered
            if result["baseline_status"] != "validated"
            and result["recovered_status"] == "validated"
        )

        regressed = sum(
            1
            for result in covered
            if result["baseline_status"] == "validated"
            and result["recovered_status"] != "validated"
        )

        field_conflicts = {}

        for result in covered:
            for conflict in result["conflicts"]:
                field_conflicts[conflict] = (
                    field_conflicts.get(conflict, 0) + 1
                )

        field_missing = {}

        for result in covered:
            for field in result.get("missing_fields", []):
                field_missing[field] = (
                    field_missing.get(field, 0) + 1
                )

        unique_mapping = sum(
            1
            for result in covered
            if result["mapping"]["method"] == "unique"
        )
        ambiguous_mapping = sum(
            1
            for result in covered
            if result["mapping"]["method"] == "ambiguous"
        )
        no_mapping = sum(
            1
            for result in covered
            if result["mapping"]["method"] == "no_candidate"
        )

        def count(status):
            return sum(
                1
                for result in covered
                if result["recovered_status"] == status
            )

        # 潜在新增可提取参数：已 confirmed 且未被任何原始参数命中的逻辑行
        matched_row_ids = {
            result["selected_logical_row_id"]
            for result in results
            if result["selected_logical_row_id"]
        }

        potential_new = []

        for row in self.rows:
            if row.get("needs_review"):
                continue

            if row.get("logical_row_id") in matched_row_ids:
                continue

            values = [
                row.get("min"),
                row.get("typ"),
                row.get("max"),
            ]

            if not any(str(value).strip() for value in values):
                continue

            potential_new.append({
                "logical_row_id": row.get("logical_row_id"),
                "parameter_symbol": row.get(
                    "parameter_symbol"
                ),
                "parameter_name": row.get(
                    "parameter_name"
                ),
                "test_conditions": row.get(
                    "test_conditions"
                ),
                "min": row.get("min"),
                "typ": row.get("typ"),
                "max": row.get("max"),
                "unit": row.get("unit"),
                "source_cell_ids": row.get(
                    "source_cell_ids", []
                ),
                "reason": (
                    "逻辑行已 confirmed，但原始 44 条参数中"
                    "没有对应参数"
                ),
            })

        return {
            "schema": SCHEMA,
            "recovered_validation_version": VERSION,
            "summary": {
                "total": total,
                "covered": len(covered),
                "not_covered": len(not_covered),
                "baseline_validated": baseline_validated,
                "baseline_validated_covered": (
                    baseline_validated_covered
                ),
                "recovered_validated": recovered_validated,
                "correctable": count("correctable"),
                "invalid": count("invalid"),
                "pending_review": count("pending_review"),
                "newly_validated": newly_validated,
                "regressed": regressed,
                "parameter_field_conflicts": field_conflicts,
                "parameter_field_conflict_total": sum(
                    field_conflicts.values()
                ),
                "parameter_field_missing": field_missing,
                "parameter_field_missing_total": sum(
                    field_missing.values()
                ),
                "evidence_mapping_coverage": round(
                    len(covered) / total, 4
                ) if total else 0.0,
                "unique_mapping_count": unique_mapping,
                "ambiguous_mapping_count": ambiguous_mapping,
                "no_mapping_count": no_mapping,
            },
            "potential_new_extractable": potential_new,
            "parameters": results,
            "notes": [
                "自动校验通过率不等于真实参数准确率。",
                "not_covered 参数因 logical_rows 未覆盖其 table_id，"
                "未参与恢复校验。",
                "confidence 仅为算法评估信息，最终以字段级证据为准。",
            ],
        }


def validate_from_files(
    extraction,
    logical_rows,
    baseline=None,
    raw_structure=None,
):
    """便捷入口：用 final_extraction（若存在）做校验。"""

    source = extraction

    if isinstance(extraction, dict) and extraction.get(
        "final_extraction"
    ):
        source = extraction["final_extraction"]

    validator = RecoveredEvidenceValidator(
        logical_rows,
        raw_structure=raw_structure,
        baseline=baseline,
    )

    return validator.validate(copy.deepcopy(source), baseline)
