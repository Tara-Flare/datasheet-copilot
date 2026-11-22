"""提取结果 JSON 保存与读取。

在不修改 DeepSeekExtractor 的前提下，
完整记录其结构化输出，供离线重校验使用。
"""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1

PARAMETER_FIELDS = (
    "name",
    "value",
    "unit",
    "value_type",
    "test_conditions",
    "evidence",
)

EVIDENCE_FIELDS = (
    "table_id",
    "row_index",
    "page",
    "source_text",
)


class ArtifactError(ValueError):
    """提取结果 JSON 不完整或格式错误。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition, message):
    if not condition:
        raise ArtifactError(message)


def validate_extraction(extraction) -> dict:
    """校验提取结果包含完整参数及 evidence 字段。"""

    _require(
        isinstance(extraction, dict),
        "提取结果必须是 JSON 对象",
    )

    parameters = extraction.get("parameters")

    _require(
        isinstance(parameters, list),
        "提取结果缺少 parameters 数组",
    )

    for index, param in enumerate(parameters):
        _require(
            isinstance(param, dict),
            f"parameters[{index}] 不是 JSON 对象",
        )

        for field in PARAMETER_FIELDS:
            _require(
                field in param,
                f"parameters[{index}] 缺少字段 {field}",
            )

        evidence = param.get("evidence")

        _require(
            isinstance(evidence, dict),
            f"parameters[{index}].evidence 不是 JSON 对象",
        )

        for field in EVIDENCE_FIELDS:
            _require(
                field in evidence,
                f"parameters[{index}].evidence 缺少字段 {field}",
            )

    return extraction


class RecordingExtractor:
    """包装现有 Extractor，记录每次结构化输出。

    不改变提取逻辑，仅在调用后保存结果副本。
    """

    def __init__(self, extractor):
        self.extractor = extractor
        self.extract_result = None
        self.retry_results = []

    def extract(self, tables):
        result = self.extractor.extract(tables)
        self.extract_result = deepcopy(result)
        return result

    def retry_parameters(self, failed_parameters, tables):
        result = self.extractor.retry_parameters(
            failed_parameters,
            tables,
        )
        self.retry_results.append(deepcopy(result))
        return result

    def get_usage(self):
        get_usage = getattr(
            self.extractor, "get_usage", None
        )

        if get_usage is None:
            return {}

        return get_usage()


def file_sha256(path) -> str:
    """计算文件 SHA-256，用于记录来源 PDF。"""

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _optional_extraction(value):
    if value is None:
        return None

    return validate_extraction(deepcopy(value))


def build_artifact(
    extraction,
    *,
    final_extraction=None,
    retry_results=None,
    pdf_path=None,
    pdf_sha256=None,
    table_cache_dir=None,
    table_cache_key=None,
    selected_table_ids=None,
    model=None,
) -> dict:
    """构造完整提取结果 artifact。

    extraction 保存 DeepSeekExtractor 的原始结构化输出；
    final_extraction 保存经过重试纠正后的最终结果。
    """

    extraction = validate_extraction(
        deepcopy(extraction)
    )

    parameters = extraction.get("parameters", [])

    if selected_table_ids is None:
        selected_table_ids = sorted({
            param["evidence"]["table_id"]
            for param in parameters
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _utc_now(),
        "source": {
            "pdf_path": (
                str(pdf_path) if pdf_path else None
            ),
            "pdf_filename": (
                Path(pdf_path).name if pdf_path else None
            ),
            "pdf_sha256": pdf_sha256,
            "table_cache_dir": (
                str(table_cache_dir)
                if table_cache_dir else None
            ),
            "table_cache_key": table_cache_key,
            "selected_table_ids": list(
                selected_table_ids
            ),
            "model": model,
        },
        "extraction": extraction,
        "final_extraction": _optional_extraction(
            final_extraction
        ),
        "retry_results": [
            validate_extraction(deepcopy(result))
            for result in (retry_results or [])
        ],
    }


def artifact_from_state(
    result,
    recorder,
    *,
    pdf_path,
    pdf_sha256=None,
    table_cache_dir=None,
    table_cache_key=None,
    model=None,
) -> dict:
    """从 E2E 最终 state 与记录器构造 artifact。"""

    raw_extraction = recorder.extract_result
    final_extraction = result.get("extraction")

    if raw_extraction is None:
        raw_extraction = final_extraction

    if raw_extraction is None:
        raise ArtifactError(
            "E2E 结果中缺少可保存的提取结果"
        )

    tables = result.get("tables", [])

    return build_artifact(
        extraction=raw_extraction,
        final_extraction=final_extraction,
        retry_results=recorder.retry_results,
        pdf_path=pdf_path,
        pdf_sha256=pdf_sha256,
        table_cache_dir=table_cache_dir,
        table_cache_key=table_cache_key,
        selected_table_ids=[
            table.get("table_id")
            for table in tables
        ],
        model=model,
    )


def save_artifact(artifact, path) -> Path:
    """原子写入提取结果 JSON。"""

    path = Path(path)

    validate_extraction(artifact.get("extraction"))

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = path.with_name(path.name + ".tmp")

        tmp.write_text(
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tmp.replace(path)

    except OSError as exc:
        raise ArtifactError(
            f"无法写入提取结果文件: {path}: {exc}"
        ) from exc

    return path


def load_artifact(path) -> dict:
    """读取并校验提取结果 JSON。"""

    path = Path(path)

    if not path.is_file():
        raise ArtifactError(
            f"提取结果文件不存在: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError) as exc:
        raise ArtifactError(
            "提取结果文件无法读取或不是合法 JSON: "
            f"{path}"
        ) from exc

    if not isinstance(data, dict):
        raise ArtifactError(
            "提取结果格式错误: 顶层必须是 JSON 对象"
        )

    schema_version = data.get("schema_version")

    if schema_version != SCHEMA_VERSION:
        raise ArtifactError(
            "不支持的提取结果 schema_version: "
            f"{schema_version}"
        )

    if "extraction" not in data:
        raise ArtifactError(
            "提取结果不完整: 缺少 extraction"
        )

    validate_extraction(data["extraction"])

    return data
