"""阶段 B2.2：RecoveredEvidenceValidator 与 LangGraph 的安全集成。

设计要点：
- EVIDENCE_VALIDATION_MODE=legacy|recovered，默认 legacy；
- legacy 模式完全保持原有行为；
- recovered 模式只对逻辑行覆盖的参数使用严格恢复校验；
- 未覆盖参数回退 legacy，但明确标记 validation_source=legacy；
- correctable 不允许映射为 validated；
- 恢复证据缺失/不匹配时明确失败或进入待审核，不静默回退。
"""

import os
from pathlib import Path

from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)


DEFAULT_MODE = "legacy"
VALID_MODES = {"legacy", "recovered"}

RAW_STRUCTURE_SCHEMA = "table_raw_structure/v1"
LOGICAL_ROWS_SCHEMA = "table_logical_rows/v1"
RAW_STRUCTURE_VERSION = 1
LOGICAL_ROWS_VERSION = 1


class RecoveredEvidenceError(ValueError):
    """恢复证据不可用或与当前 PDF 不匹配。"""


def resolve_evidence_validation_mode(value=None):
    """解析校验模式；只允许显式 legacy / recovered。"""

    mode = (
        value
        or os.getenv("EVIDENCE_VALIDATION_MODE", DEFAULT_MODE)
    )
    mode = str(mode).strip().lower()

    if mode not in VALID_MODES:
        raise ValueError(
            "不支持的 EVIDENCE_VALIDATION_MODE: "
            f"{mode!r}（只允许 legacy 或 recovered）"
        )

    return mode


def _raw_tables(raw_structure):
    return {
        table.get("table_id"): table
        for table in raw_structure.get("tables", [])
    }


def check_recovered_source_consistency(config):
    """检查 logical_rows / raw_structure / PDF 是否属于同一来源。"""

    errors = []

    logical_rows = config.get("logical_rows")
    raw_structure = config.get("raw_structure")

    if not isinstance(logical_rows, dict):
        errors.append("logical_rows 缺失或不是 JSON 对象")
    elif not logical_rows.get("logical_rows"):
        errors.append("logical_rows 为空")

    if not isinstance(raw_structure, dict):
        errors.append("raw_structure 缺失或不是 JSON 对象")
    elif not raw_structure.get("tables"):
        errors.append("raw_structure 为空")

    if errors:
        return {"ok": False, "errors": errors}

    if raw_structure.get("schema") != RAW_STRUCTURE_SCHEMA:
        errors.append(
            "不支持的 raw_structure schema: "
            f"{raw_structure.get('schema')!r}"
        )

    if (
        raw_structure.get("raw_structure_version")
        != RAW_STRUCTURE_VERSION
    ):
        errors.append(
            "不支持的 raw_structure_version: "
            f"{raw_structure.get('raw_structure_version')!r}"
        )

    if logical_rows.get("schema") != LOGICAL_ROWS_SCHEMA:
        errors.append(
            "不支持的 logical_rows schema: "
            f"{logical_rows.get('schema')!r}"
        )

    if (
        logical_rows.get("logical_rows_version")
        != LOGICAL_ROWS_VERSION
    ):
        errors.append(
            "不支持的 logical_rows_version: "
            f"{logical_rows.get('logical_rows_version')!r}"
        )

    expected_sha = (
        raw_structure.get("source") or {}
    ).get("pdf_sha256")

    if not expected_sha:
        errors.append("raw_structure 缺少 PDF SHA256")

    actual_sha = config.get("pdf_sha256")

    if (
        expected_sha
        and actual_sha
        and expected_sha != actual_sha
    ):
        errors.append(
            "PDF SHA256 不匹配: "
            f"expected={expected_sha}, actual={actual_sha}"
        )

    tables = _raw_tables(raw_structure)

    for row in logical_rows.get("logical_rows", []):
        row_id = row.get("logical_row_id")
        table_id = row.get("table_id")
        table = tables.get(table_id)

        if table is None:
            errors.append(
                f"{row_id} 引用的 table_id={table_id} "
                "不在 raw_structure 中"
            )
            continue

        cell_ids = {
            cell.get("cell_id")
            for cell in table.get("cells", [])
        }

        missing_cells = [
            cell_id
            for cell_id in row.get("source_cell_ids", [])
            if cell_id not in cell_ids
        ]

        if missing_cells:
            errors.append(
                f"{row_id} 的 source_cell_ids 不在 raw_structure: "
                f"{missing_cells}"
            )

        page = row.get("page")

        if page not in (table.get("page_numbers") or []):
            errors.append(
                f"{row_id} 的 page={page} 与 raw_structure "
                f"page_numbers={table.get('page_numbers')} 不一致"
            )

        num_rows = int(table.get("num_rows") or 0)

        for row_index in row.get("source_row_indices", []):
            if (
                not isinstance(row_index, int)
                or row_index < 0
                or row_index >= max(num_rows, 1)
            ):
                errors.append(
                    f"{row_id} 的 source_row_index="
                    f"{row_index} 超出范围"
                )

    return {"ok": not errors, "errors": errors}


def adapt_recovered_status(status):
    """把 detailed_recovery_status 映射为 Agent validation_status。

    correctable 不能映射为 validated。
    """

    if status == "validated":
        return "validated"

    if status == "invalid":
        return "invalid"

    return "pending_review"


def _errors_from_recovered(recovered):
    errors = []

    status = recovered.get("recovered_status")

    if status == "invalid":
        for conflict in recovered.get("conflicts", []):
            errors.append(f"recovered_conflict:{conflict}")

        conditions = recovered.get("field_checks", {}).get(
            "test_conditions", {}
        )

        for item in conditions.get(
            "conflicting_conditions", []
        ):
            errors.append(f"条件冲突: {item}")

    elif status == "correctable":
        for field in recovered.get("missing_fields", []):
            errors.append(f"缺少字段: {field}")

        conditions = recovered.get("field_checks", {}).get(
            "test_conditions", {}
        )

        for item in conditions.get(
            "missing_conditions", []
        ):
            errors.append(f"缺少条件: {item}")

    elif status == "pending_review":
        errors.append(
            recovered.get("reason") or "recovered_pending_review"
        )

    return errors


def _unavailable_result(extraction, errors):
    parameters = []

    for param in extraction.get("parameters", []):
        merged = dict(param)
        merged["validation_status"] = "pending_review"
        merged["validation_errors"] = [
            "恢复证据不可用: " + "; ".join(errors)
        ]
        merged["validation_source"] = "recovered"
        merged["detailed_recovery_status"] = "unavailable"
        merged["recovered_logical_row_id"] = None
        merged["identity_ambiguous"] = False
        merged["recovered_reason"] = None
        merged["missing_fields"] = []
        merged["conflicting_fields"] = []
        merged["missing_conditions"] = []
        merged["conflicting_conditions"] = []
        merged["suggested_correction"] = None
        parameters.append(merged)

    summary = {
        "total": len(parameters),
        "validated": 0,
        "invalid": 0,
        "pending_review": len(parameters),
        "validation_mode": "recovered",
        "recovered_validated": 0,
        "legacy_validated": 0,
        "correctable": 0,
        "recovered_evidence_error": list(errors),
    }

    return {
        "parameters": parameters,
        "validation_summary": summary,
        "validation_mode": "recovered",
        "recovered_validation": {
            "available": False,
            "errors": list(errors),
        },
    }


def _merge_results(extraction, legacy_result, recovered_report):
    legacy_params = legacy_result.get("parameters", [])
    recovered_params = recovered_report.get("parameters", [])

    merged_parameters = []

    for index, param in enumerate(
        extraction.get("parameters", [])
    ):
        legacy_param = (
            legacy_params[index]
            if index < len(legacy_params)
            else {}
        )
        recovered = (
            recovered_params[index]
            if index < len(recovered_params)
            else {}
        )

        merged = dict(param)
        status = recovered.get("recovered_status")

        if status == "not_covered":
            merged["validation_status"] = legacy_param.get(
                "validation_status", "pending_review"
            )
            merged["validation_errors"] = list(
                legacy_param.get("validation_errors", [])
            )
            merged["validation_source"] = "legacy"
            merged["detailed_recovery_status"] = "not_covered"
            merged["recovered_logical_row_id"] = None
            merged["identity_ambiguous"] = False
            merged["recovered_reason"] = None
            merged["missing_fields"] = []
            merged["conflicting_fields"] = []
            merged["missing_conditions"] = []
            merged["conflicting_conditions"] = []
            merged["suggested_correction"] = None
        else:
            merged["validation_status"] = (
                adapt_recovered_status(status)
            )
            merged["validation_errors"] = (
                _errors_from_recovered(recovered)
            )
            merged["validation_source"] = "recovered"
            merged["detailed_recovery_status"] = status
            merged["recovered_logical_row_id"] = recovered.get(
                "selected_logical_row_id"
            )
            merged["identity_ambiguous"] = bool(
                recovered.get("identity_ambiguous")
            )
            merged["recovered_reason"] = recovered.get("reason")
            merged["missing_fields"] = recovered.get(
                "missing_fields", []
            )
            merged["conflicting_fields"] = recovered.get(
                "conflicts", []
            )

            conditions = recovered.get(
                "field_checks", {}
            ).get("test_conditions", {})

            merged["missing_conditions"] = conditions.get(
                "missing_conditions", []
            )
            merged["conflicting_conditions"] = conditions.get(
                "conflicting_conditions", []
            )

            suggestions = recovered.get(
                "correction_suggestions", []
            )
            merged["suggested_correction"] = (
                suggestions[0] if suggestions else None
            )

        merged_parameters.append(merged)

    summary = _build_summary(
        merged_parameters, recovered_report
    )

    return {
        "parameters": merged_parameters,
        "validation_summary": summary,
        "validation_mode": "recovered",
        "recovered_validation": {
            "available": True,
            "summary": recovered_report.get("summary", {}),
            "unresolved_issues": recovered_report.get(
                "unresolved_issues", []
            ),
        },
    }


def _build_summary(parameters, recovered_report):
    def status_count(status):
        return sum(
            1
            for param in parameters
            if param.get("validation_status") == status
        )

    recovered_validated = sum(
        1
        for param in parameters
        if param.get("validation_source") == "recovered"
        and param.get("validation_status") == "validated"
    )

    legacy_validated = sum(
        1
        for param in parameters
        if param.get("validation_source") == "legacy"
        and param.get("validation_status") == "validated"
    )

    correctable = sum(
        1
        for param in parameters
        if param.get("detailed_recovery_status")
        == "correctable"
    )

    not_covered = sum(
        1
        for param in parameters
        if param.get("detailed_recovery_status")
        == "not_covered"
    )

    recovered_invalid = sum(
        1
        for param in parameters
        if param.get("validation_source") == "recovered"
        and param.get("validation_status") == "invalid"
    )

    recovered_pending = sum(
        1
        for param in parameters
        if param.get("validation_source") == "recovered"
        and param.get("validation_status") == "pending_review"
    )

    return {
        "total": len(parameters),
        "validated": status_count("validated"),
        "invalid": status_count("invalid"),
        "pending_review": status_count("pending_review"),
        "validation_mode": "recovered",
        "recovered_validated": recovered_validated,
        "legacy_validated": legacy_validated,
        "correctable": correctable,
        "not_covered": not_covered,
        "recovered_invalid": recovered_invalid,
        "recovered_pending_review": recovered_pending,
        "recovered_evidence_error": None,
        "recovered_evidence_summary": recovered_report.get(
            "summary", {}
        ),
    }


def build_recovered_validation_result(
    extraction,
    legacy_result,
    config,
):
    """构建 recovered 模式的合并校验结果。

    参数：
    - extraction: 当前参数提取结果
    - legacy_result: 已执行的 legacy 校验结果
    - config: {logical_rows, raw_structure, pdf_sha256, on_error}
    """

    consistency = check_recovered_source_consistency(
        config or {}
    )

    if not consistency["ok"]:
        on_error = (config or {}).get(
            "on_error", "pending_review"
        )

        if on_error == "fail":
            raise RecoveredEvidenceError(
                "恢复证据不可用或与当前 PDF 不匹配: "
                + "; ".join(consistency["errors"])
            )

        return _unavailable_result(
            extraction, consistency["errors"]
        )

    recovered_validator = RecoveredEvidenceValidator(
        config["logical_rows"],
        raw_structure=config["raw_structure"],
        baseline=legacy_result,
    )

    recovered_report = recovered_validator.validate(
        extraction,
        baseline=legacy_result,
    )

    return _merge_results(
        extraction, legacy_result, recovered_report
    )


def load_recovered_evidence_config(
    logical_rows_path,
    raw_structure_path,
    *,
    pdf_sha256=None,
    on_error="pending_review",
):
    """从文件加载 recovered 证据配置（供 E2E 脚本使用）。"""

    import json

    logical_rows = json.loads(
        Path(logical_rows_path).read_text(encoding="utf-8")
    )
    raw_structure = json.loads(
        Path(raw_structure_path).read_text(encoding="utf-8")
    )

    return {
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "pdf_sha256": pdf_sha256,
        "on_error": on_error,
    }
