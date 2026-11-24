"""第一版 MVP 生产流水线装配。

默认方案 D（Logical + P1 + recovered 校验），
可配置回退方案 C（Flat + P1 + legacy 校验）。

设计约束：
- 生产入口必须真实运行 StructureRecovery / LogicalTableAdapter，
  不能只改一个配置名称；
- 结构化恢复只在目标表格上执行，恢复失败时不会把 Flat 表
  静默送进 D 方案；
- validated 才是自动接受；pending_review 单独展示；
  invalid 阻止自动接受；
- 不修改冻结 Gold，也不改变历史实验产物。
"""

import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agents.graph import build_graph
from app.extraction.deepseek_extractor import DeepSeekExtractor
from app.parsers.pdf_parser import PDFParser
from app.parsers.structure_recovery import (
    LOGICAL_ROWS_VERSION,
    SCHEMA as LOGICAL_ROWS_SCHEMA,
    recover_from_pdf,
)
from app.parsers.table_parser import TableParser
from app.parsers.table_selector import TableSelector


logger = logging.getLogger(__name__)


DEFAULT_PROFILE = "d"
DEFAULT_MAX_TABLES = 4
DEFAULT_MAX_CHARS = 40000
DEFAULT_DOCLING_ARTIFACTS = (
    Path(__file__).resolve().parents[2] / "docling-models"
)


def configure_docling_environment():
    """离线部署时默认复用仓库内 Docling 模型目录。"""

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if (
        not os.getenv("DOCLING_ARTIFACTS_PATH")
        and DEFAULT_DOCLING_ARTIFACTS.is_dir()
    ):
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(
            DEFAULT_DOCLING_ARTIFACTS
        )


class PipelineError(ValueError):
    """生产流水线配置或运行前置条件不满足。"""


@dataclass(frozen=True)
class RunProfile:
    """MVP 运行方案。"""

    key: str
    label: str
    input_mode: str
    family_coverage: bool
    evidence_validation_mode: str
    prompt_family: str

    def to_dict(self):
        return asdict(self)


PROFILES = {
    "d": RunProfile(
        key="d",
        label="Logical + P1",
        input_mode="logical",
        family_coverage=True,
        evidence_validation_mode="recovered",
        prompt_family="P1",
    ),
    "c": RunProfile(
        key="c",
        label="Flat + P1",
        input_mode="flat",
        family_coverage=True,
        evidence_validation_mode="legacy",
        prompt_family="P1",
    ),
}


def resolve_run_profile(value=None):
    """解析 DATASHEET_RUN_PROFILE；默认 d，可显式选择 c。"""

    raw = (
        value
        if value is not None
        else os.getenv("DATASHEET_RUN_PROFILE", DEFAULT_PROFILE)
    )
    key = str(raw).strip().lower()

    if key not in PROFILES:
        raise PipelineError(
            "不支持的 DATASHEET_RUN_PROFILE: "
            f"{raw!r}（只允许 d 或 c）"
        )

    return PROFILES[key]


def parse_table_ids(value):
    """解析逗号分隔的 table_id 列表。"""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    text = str(value).strip()

    if not text:
        return None

    return [
        int(item)
        for item in text.split(",")
        if item.strip()
    ]


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class PrecomputedParser:
    """返回已解析结果，避免同一 PDF 被 Docling 重复解析。"""

    def __init__(self, payload, name="parser"):
        self.payload = payload
        self.name = name

    def parse(self, pdf_path):
        return deepcopy(self.payload)


def _selected_tables(tables, table_ids):
    if table_ids is None:
        return list(tables)

    requested = [int(value) for value in table_ids]
    by_id = {
        table.get("table_id"): table
        for table in tables
    }
    missing = [
        table_id for table_id in requested
        if table_id not in by_id
    ]

    if missing:
        raise PipelineError(
            "请求的 table_id 不存在于解析结果: "
            f"{missing}"
        )

    return [by_id[table_id] for table_id in requested]


def _recover_table(raw_structure, table, pdf_path):
    table_id = table.get("table_id")
    page_numbers = table.get("page_numbers") or []

    if not page_numbers:
        raise PipelineError(
            f"table_id={table_id} 缺少 page_numbers，无法恢复结构"
        )

    page_no = int(page_numbers[0])

    return recover_from_pdf(
        raw_structure,
        table_id,
        str(pdf_path),
        page_no,
    )


def _merge_logical_rows(recovered_tables):
    rows = []

    for recovered in recovered_tables:
        rows.extend(recovered.get("logical_rows", []))

    return {
        "schema": LOGICAL_ROWS_SCHEMA,
        "logical_rows_version": LOGICAL_ROWS_VERSION,
        "tables": recovered_tables,
        "logical_rows": rows,
    }


def build_runtime_inputs(
    pdf_path,
    *,
    profile=None,
    pdf_parser=None,
    table_parser=None,
    selector=None,
    table_ids=None,
    logical_rows_override=None,
    raw_structure_override=None,
):
    """真实解析 PDF 并在 Logical 方案下运行 StructureRecovery。

    返回 graph 需要的 logical_extraction / recovered_evidence
    以及可审计的运行时诊断信息。
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise PipelineError(f"PDF 文件不存在: {pdf_path}")

    resolved = (
        profile
        if isinstance(profile, RunProfile)
        else resolve_run_profile(profile)
    )

    pdf_parser = pdf_parser or PDFParser()
    table_parser = table_parser or TableParser()
    selector = selector or TableSelector(
        max_tables=DEFAULT_MAX_TABLES,
        max_chars=DEFAULT_MAX_CHARS,
    )

    pdf_document = pdf_parser.parse(str(pdf_path))
    table_result = table_parser.parse(
        str(pdf_path),
        include_raw_structure=True,
    )

    all_tables = table_result.get("tables", [])
    selected = selector.select(all_tables)

    if table_ids is not None:
        selected = _selected_tables(selected, table_ids)
        selection = "explicit"
    else:
        selection = "selector"

    raw_structure = (
        raw_structure_override
        if raw_structure_override is not None
        else table_result.get("raw_structure")
    )

    if not isinstance(raw_structure, dict):
        raise PipelineError(
            "TableParser 未返回 raw_structure，"
            "无法运行 StructureRecovery"
        )

    diagnostics = {
        "profile": resolved.to_dict(),
        "table_selection": {
            "mode": selection,
            "requested_table_ids": (
                [int(value) for value in table_ids]
                if table_ids is not None
                else None
            ),
            "selected_table_ids": [
                table.get("table_id") for table in selected
            ],
            "available_table_ids": [
                table.get("table_id") for table in all_tables
            ],
        },
        "structure_recovery": {
            "enabled": resolved.input_mode == "logical",
            "attempted": 0,
            "recovered": 0,
            "failed": [],
            "logical_table_ids": [],
            "logical_row_count": 0,
            "source": (
                "override"
                if logical_rows_override is not None
                else "runtime"
            ),
        },
    }

    if resolved.input_mode != "logical":
        diagnostics["structure_recovery"]["enabled"] = False

        return {
            "profile": resolved,
            "pdf_document": pdf_document,
            "all_tables": all_tables,
            "table_result": {
                "filename": pdf_path.name,
                "table_count": len(selected),
                "original_table_count": len(all_tables),
                "tables": selected,
                "status": "parsed",
            },
            "raw_structure": raw_structure,
            "logical_rows": None,
            "logical_extraction": None,
            "recovered_evidence": None,
            "diagnostics": diagnostics,
        }

    if logical_rows_override is not None:
        logical_rows = logical_rows_override
        recovered_ids = [
            group.get("table_id")
            for group in logical_rows.get("tables", [])
        ]
        if not recovered_ids:
            recovered_ids = [
                row.get("table_id")
                for row in logical_rows.get("logical_rows", [])
                if row.get("table_id") is not None
            ]
        diagnostics["structure_recovery"][
            "logical_table_ids"
        ] = sorted(set(recovered_ids))
        diagnostics["structure_recovery"]["recovered"] = len(
            set(recovered_ids)
        )
        diagnostics["structure_recovery"]["logical_row_count"] = len(
            logical_rows.get("logical_rows", [])
        )
        selected = [
            table for table in selected
            if table.get("table_id") in set(recovered_ids)
        ]
    else:
        recovered_tables = []
        failed = []

        for table in selected:
            diagnostics["structure_recovery"]["attempted"] += 1
            table_id = table.get("table_id")

            try:
                recovered = _recover_table(
                    raw_structure, table, pdf_path
                )
            except Exception as exc:  # noqa: BLE001
                failed.append({
                    "table_id": table_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                logger.warning(
                    "StructureRecovery 失败: table_id=%s: %s",
                    table_id,
                    exc,
                )
                continue

            recovered_tables.append(recovered)

        diagnostics["structure_recovery"]["failed"] = failed
        diagnostics["structure_recovery"]["recovered"] = len(
            recovered_tables
        )
        diagnostics["structure_recovery"]["logical_table_ids"] = [
            item.get("table_id") for item in recovered_tables
        ]
        diagnostics["structure_recovery"]["logical_row_count"] = sum(
            len(item.get("logical_rows", []))
            for item in recovered_tables
        )

        if not recovered_tables:
            raise PipelineError(
                "Logical 方案要求 StructureRecovery 成功，"
                "但没有任何目标表恢复成功；"
                f"失败详情: {failed}"
            )

        if table_ids is not None and failed:
            raise PipelineError(
                "显式指定的目标表结构恢复失败，"
                "拒绝降级为 Flat 自动接受: "
                f"{failed}"
            )

        recovered_ids = [
            item.get("table_id") for item in recovered_tables
        ]
        selected = [
            table for table in selected
            if table.get("table_id") in set(recovered_ids)
        ]
        logical_rows = _merge_logical_rows(recovered_tables)

    pdf_sha256 = (
        (raw_structure.get("source") or {}).get("pdf_sha256")
        or file_sha256(pdf_path)
    )

    logical_extraction = {
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "table_ids": sorted(set(recovered_ids)),
        "include_needs_review": False,
    }
    recovered_evidence = {
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "pdf_sha256": pdf_sha256,
        "on_error": "pending_review",
    }

    return {
        "profile": resolved,
        "pdf_document": pdf_document,
        "all_tables": all_tables,
        "table_result": {
            "filename": pdf_path.name,
            "table_count": len(selected),
            "original_table_count": len(all_tables),
            "tables": selected,
            "status": "parsed",
        },
        "raw_structure": raw_structure,
        "logical_rows": logical_rows,
        "logical_extraction": logical_extraction,
        "recovered_evidence": recovered_evidence,
        "diagnostics": diagnostics,
    }


def summarize_validation(validation):
    """把校验结果拆成自动接受 / 待复核 / 拒绝三组。"""

    parameters = validation.get("parameters", [])
    validated = [
        item for item in parameters
        if item.get("validation_status") == "validated"
    ]
    pending = [
        item for item in parameters
        if item.get("validation_status") == "pending_review"
    ]
    invalid = [
        item for item in parameters
        if item.get("validation_status") == "invalid"
    ]

    return {
        "total": len(parameters),
        "validated": len(validated),
        "pending_review": len(pending),
        "invalid": len(invalid),
        "auto_accepted": len(validated),
        "auto_accepted_parameters": validated,
        "pending_parameters": pending,
        "invalid_parameters": invalid,
    }


def run_pipeline(
    pdf_path,
    *,
    profile=None,
    extractor=None,
    pdf_parser=None,
    table_parser=None,
    selector=None,
    table_ids=None,
    logical_rows_override=None,
    raw_structure_override=None,
    max_retries=1,
    enable_human_review=False,
    checkpointer=None,
):
    """执行真实生产链路，返回 state、运行诊断与 extractor。"""

    runtime = build_runtime_inputs(
        pdf_path,
        profile=profile,
        pdf_parser=pdf_parser,
        table_parser=table_parser,
        selector=selector,
        table_ids=table_ids,
        logical_rows_override=logical_rows_override,
        raw_structure_override=raw_structure_override,
    )
    resolved = runtime["profile"]

    if extractor is None:
        extractor = DeepSeekExtractor(
            family_coverage=resolved.family_coverage,
        )

    graph = build_graph(
        pdf_parser=PrecomputedParser(
            runtime["pdf_document"], "pdf"
        ),
        table_parser=PrecomputedParser(
            runtime["table_result"], "table"
        ),
        extractor=extractor,
        checkpointer=checkpointer,
        evidence_validation_mode=(
            resolved.evidence_validation_mode
        ),
        recovered_evidence=runtime["recovered_evidence"],
        extraction_input_mode=resolved.input_mode,
        logical_extraction=runtime["logical_extraction"],
    )

    state = graph.invoke(
        {
            "document_path": str(Path(pdf_path).resolve()),
            "status": "pending",
            "retry_count": 0,
            "max_retries": max_retries,
            "enable_human_review": enable_human_review,
            "errors": [],
        },
        config={
            "configurable": {
                "thread_id": os.urandom(16).hex(),
            }
        },
    )

    diagnostics = dict(runtime["diagnostics"])
    diagnostics["effective_input_mode"] = state.get(
        "extraction_input_mode"
    )
    diagnostics["input_summary"] = state.get(
        "extraction_input_summary"
    )
    diagnostics["validation_mode"] = state.get(
        "evidence_validation_mode"
    )

    validation = state.get("validation_result", {})

    return {
        "state": state,
        "runtime": runtime,
        "diagnostics": diagnostics,
        "validation": validation,
        "summary": summarize_validation(validation),
        "pdf_document": runtime["pdf_document"],
        "extractor": extractor,
        "profile": resolved,
    }


def dumps_runtime(payload):
    """便于脚本写出运行诊断。"""

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
