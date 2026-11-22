"""离线重校验：只读取提取结果与 Docling 缓存。

不调用 DeepSeek API，不重新解析 PDF。
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.extraction.artifact_store import (
    ArtifactError,
    validate_extraction,
)
from app.extraction.evidence import verify_evidence
from app.validation.validator import ParameterValidator


REPORT_SCHEMA_VERSION = 1

DEFAULT_VALIDATOR_VERSION = "evidence+parameter"

EXTRACTION_SOURCES = (
    "extraction",
    "final_extraction",
)


class OfflineValidationError(ValueError):
    """离线校验输入缺失或不完整。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validator_fingerprint() -> str:
    """计算 Validator 相关源码指纹，用于区分版本。"""

    base = Path(__file__).resolve().parent
    app_dir = base.parent

    files = [
        base / "validator.py",
        base / "rules.py",
        base / "units.py",
        app_dir / "extraction" / "evidence.py",
    ]

    digest = hashlib.sha256()

    for path in files:
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(str(path).encode("utf-8"))

    return digest.hexdigest()


def load_cached_tables(cache_path) -> list[dict]:
    """读取并校验 Docling 表格缓存。"""

    path = Path(cache_path)

    if not path.is_file():
        raise OfflineValidationError(
            f"表格缓存文件不存在: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError) as exc:
        raise OfflineValidationError(
            "表格缓存文件无法读取或不是合法 JSON: "
            f"{path}"
        ) from exc

    if not isinstance(data, dict):
        raise OfflineValidationError(
            "表格缓存格式错误: 顶层必须是 JSON 对象"
        )

    tables = data.get("tables")

    if not isinstance(tables, list):
        raise OfflineValidationError(
            "表格缓存不完整: 缺少 tables 数组"
        )

    if not tables:
        raise OfflineValidationError(
            "表格缓存不完整: tables 为空"
        )

    for index, table in enumerate(tables):
        if not isinstance(table, dict):
            raise OfflineValidationError(
                f"表格缓存不完整: tables[{index}] "
                "不是 JSON 对象"
            )

        if "table_id" not in table:
            raise OfflineValidationError(
                f"表格缓存不完整: tables[{index}] "
                "缺少 table_id"
            )

        rows = table.get("rows")

        if not isinstance(rows, list):
            raise OfflineValidationError(
                f"表格缓存不完整: tables[{index}] "
                "缺少 rows 数组"
            )

    return tables


def select_extraction(
    artifact,
    source="extraction",
) -> dict:
    """从 artifact 中选择要校验的提取结果。"""

    if source not in EXTRACTION_SOURCES:
        raise OfflineValidationError(
            f"不支持的提取结果来源: {source}"
        )

    extraction = artifact.get(source)

    if extraction is None:
        raise OfflineValidationError(
            f"提取结果中不存在 {source}，"
            "无法离线校验"
        )

    try:
        return validate_extraction(extraction)

    except ArtifactError as exc:
        raise OfflineValidationError(
            str(exc)
        ) from exc


def referenced_table_ids(extraction) -> set:
    """返回参数引用到的全部 table_id。"""

    table_ids = set()

    for param in extraction.get("parameters", []):
        evidence = param.get("evidence") or {}
        table_id = evidence.get("table_id")

        if table_id is not None:
            table_ids.add(table_id)

    return table_ids


def ensure_tables_available(tables, extraction) -> None:
    """确认缓存包含参数引用的全部表格。"""

    available = {
        table.get("table_id")
        for table in tables
        if isinstance(table, dict)
    }

    missing = referenced_table_ids(
        extraction
    ) - available

    if missing:
        raise OfflineValidationError(
            "表格缓存不完整: 缺少参数引用的表格 "
            f"table_id={sorted(missing)}"
        )


def revalidate(extraction, tables) -> dict:
    """重新执行 EvidenceValidator 与 ParameterValidator。"""

    evidence_result = verify_evidence(
        extraction,
        tables,
    )

    return ParameterValidator().validate(
        evidence_result
    )


def categorize_error(message: str) -> str:
    """将错误信息归入稳定类别，便于统计对比。"""

    if message == "原文引用不匹配":
        return "原文引用"

    if message.startswith("提取数值"):
        return "数值"

    if message.startswith("提取单位"):
        return "单位"

    if "来源" in message:
        return "来源定位"

    if "大于" in message and any(
        key in message
        for key in ("min", "typ", "max")
    ):
        return "Min/Typ/Max 关系"

    return "其他"


def build_validation_report(
    extraction,
    tables,
    *,
    validator_version=None,
    extraction_source="extraction",
    artifact_path=None,
    extraction_metadata=None,
) -> dict:
    """执行离线校验并构造可对比的结果报告。"""

    result = revalidate(extraction, tables)

    parameters = result["parameters"]

    by_message = {}
    by_category = {}
    total_errors = 0
    parameters_with_errors = 0

    for param in parameters:
        errors = param.get("validation_errors", [])

        if errors:
            parameters_with_errors += 1

        for message in errors:
            total_errors += 1

            by_message[message] = (
                by_message.get(message, 0) + 1
            )

            category = categorize_error(message)

            by_category[category] = (
                by_category.get(category, 0) + 1
            )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validated_at": _utc_now(),
        "validator_version": (
            validator_version
            or DEFAULT_VALIDATOR_VERSION
        ),
        "validator_fingerprint": validator_fingerprint(),
        "extraction_source": extraction_source,
        "source_artifact": (
            str(artifact_path)
            if artifact_path else None
        ),
        "extraction_metadata": extraction_metadata,
        "summary": result["validation_summary"],
        "error_statistics": {
            "total_errors": total_errors,
            "parameters_with_errors": (
                parameters_with_errors
            ),
            "by_message": by_message,
            "by_category": by_category,
        },
        "parameters": parameters,
    }


def save_validation_report(report, path) -> Path:
    """原子写入校验结果 JSON。"""

    path = Path(path)

    if not isinstance(report, dict) or not isinstance(
        report.get("parameters"), list
    ):
        raise OfflineValidationError(
            "校验结果不完整: 缺少 parameters 数组"
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = path.with_name(path.name + ".tmp")

        tmp.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tmp.replace(path)

    except OSError as exc:
        raise OfflineValidationError(
            f"无法写入校验结果文件: {path}: {exc}"
        ) from exc

    return path


def load_validation_report(path) -> dict:
    """读取校验结果 JSON。"""

    path = Path(path)

    if not path.is_file():
        raise OfflineValidationError(
            f"校验结果文件不存在: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError) as exc:
        raise OfflineValidationError(
            "校验结果文件无法读取或不是合法 JSON: "
            f"{path}"
        ) from exc

    if not isinstance(data, dict) or not isinstance(
        data.get("parameters"), list
    ):
        raise OfflineValidationError(
            f"校验结果文件不完整: {path}"
        )

    return data


def _classify_change(old_status, new_status) -> str:
    if old_status != "validated" and new_status == "validated":
        return "improved"

    if old_status == "validated" and new_status != "validated":
        return "regressed"

    return "changed"


def compare_reports(old, new) -> dict:
    """对比两份校验报告。

    两份报告应对同一份提取结果生成，
    因此按参数下标对齐。
    """

    old_params = old.get("parameters", [])
    new_params = new.get("parameters", [])

    changes = []

    for index in range(
        min(len(old_params), len(new_params))
    ):
        old_param = old_params[index]
        new_param = new_params[index]

        old_status = old_param.get(
            "validation_status"
        )
        new_status = new_param.get(
            "validation_status"
        )

        old_errors = old_param.get(
            "validation_errors", []
        )
        new_errors = new_param.get(
            "validation_errors", []
        )

        if (
            old_status == new_status
            and old_errors == new_errors
        ):
            continue

        changes.append({
            "index": index,
            "name": (
                new_param.get("name")
                or old_param.get("name")
            ),
            "value_type": (
                new_param.get("value_type")
                or old_param.get("value_type")
            ),
            "test_conditions": (
                new_param.get("test_conditions")
                or old_param.get("test_conditions")
                or ""
            ),
            "old_status": old_status,
            "new_status": new_status,
            "old_errors": old_errors,
            "new_errors": new_errors,
            "change_type": _classify_change(
                old_status,
                new_status,
            ),
        })

    improved = sum(
        1
        for change in changes
        if change["change_type"] == "improved"
    )

    regressed = sum(
        1
        for change in changes
        if change["change_type"] == "regressed"
    )

    return {
        "old_validator_version": old.get(
            "validator_version"
        ),
        "new_validator_version": new.get(
            "validator_version"
        ),
        "old_validator_fingerprint": old.get(
            "validator_fingerprint"
        ),
        "new_validator_fingerprint": new.get(
            "validator_fingerprint"
        ),
        "old_summary": old.get("summary"),
        "new_summary": new.get("summary"),
        "parameter_count_old": len(old_params),
        "parameter_count_new": len(new_params),
        "improved": improved,
        "regressed": regressed,
        "changed_count": len(changes),
        "changes": changes,
    }
