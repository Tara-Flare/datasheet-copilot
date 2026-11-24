"""Datasheet Copilot 真实端到端提参入口（MVP）。

默认方案：
- D：Logical + P1 + recovered 校验（生产默认）
可配置回退：
- C：Flat + P1 + legacy 校验

生产链路会真实解析 PDF、按需运行 StructureRecovery，
再通过 LogicalTableAdapter 装配模型输入，最后执行
legacy + recovered 校验。不会写死参数值。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (PROJECT_ROOT / "backend",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.extraction.artifact_store import (  # noqa: E402
    RecordingExtractor,
    artifact_from_state,
    file_sha256,
    save_artifact,
)
from app.extraction.deepseek_extractor import (  # noqa: E402
    DeepSeekExtractor,
)
from app.pipeline import (  # noqa: E402
    PipelineError,
    parse_table_ids,
    resolve_run_profile,
    run_pipeline,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/outputs"
DEFAULT_DOCLING_ARTIFACTS = PROJECT_ROOT / "docling-models"


def default_output_path(pdf_path: str) -> Path:
    """默认提取结果保存路径。"""

    return (
        DEFAULT_OUTPUT_DIR
        / f"{Path(pdf_path).stem}.extraction.json"
    )


def configure_docling_environment():
    """离线环境下复用仓库内 Docling 模型。"""

    if (
        not os.getenv("DOCLING_ARTIFACTS_PATH")
        and DEFAULT_DOCLING_ARTIFACTS.is_dir()
    ):
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(
            DEFAULT_DOCLING_ARTIFACTS
        )


def _load_override(path):
    if not path:
        return None

    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_e2e(
    pdf_path: str,
    output_json=None,
    *,
    profile=None,
    table_ids=None,
):
    """运行真实 Datasheet 端到端提参流程。"""

    configure_docling_environment()

    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).is_file():
        raise FileNotFoundError(
            f"找不到 PDF 文件: {pdf_path}"
        )

    resolved = resolve_run_profile(profile)

    logical_rows_override = _load_override(
        os.getenv("RECOVERED_EVIDENCE_LOGICAL_ROWS")
    )
    raw_structure_override = _load_override(
        os.getenv("RECOVERED_EVIDENCE_RAW_STRUCTURE")
    )

    if logical_rows_override is not None and raw_structure_override is None:
        raise PipelineError(
            "设置 RECOVERED_EVIDENCE_LOGICAL_ROWS 时必须同时设置 "
            "RECOVERED_EVIDENCE_RAW_STRUCTURE"
        )

    # 先完成配置校验（例如缺少 LLM_API_KEY），避免先打印运行横幅再报错。
    extractor = DeepSeekExtractor(
        family_coverage=resolved.family_coverage,
    )
    recording_extractor = RecordingExtractor(extractor)

    print("=" * 60)
    print("Datasheet Copilot - E2E")
    print("=" * 60)
    print(f"PDF 文件: {pdf_path}")
    print(f"运行方案: {resolved.key.upper()} ({resolved.label})")
    print(
        "输入模式:",
        resolved.input_mode,
        "| Prompt:",
        resolved.prompt_family,
        "| 校验:",
        resolved.evidence_validation_mode,
    )

    started = time.perf_counter()

    run = run_pipeline(
        pdf_path,
        profile=resolved,
        extractor=recording_extractor,
        table_ids=table_ids,
        logical_rows_override=logical_rows_override,
        raw_structure_override=raw_structure_override,
    )

    duration = time.perf_counter() - started

    result = run["state"]
    validation = run["validation"]
    summary = run["summary"]
    diagnostics = run["diagnostics"]

    if output_json is None:
        output_json = default_output_path(pdf_path)

    artifact = artifact_from_state(
        result,
        recording_extractor,
        pdf_path=pdf_path,
        pdf_sha256=file_sha256(pdf_path),
        model=getattr(extractor, "model", None),
    )
    saved_path = save_artifact(artifact, output_json)

    sidecar = Path(saved_path).with_suffix(
        ".pipeline.json"
    )
    _write_json(
        sidecar,
        {
            "schema": "mvp_pipeline_run/v1",
            "pdf_path": pdf_path,
            "profile": resolved.to_dict(),
            "diagnostics": diagnostics,
            "validation_summary": summary,
            "usage": extractor.get_usage(),
            "elapsed_s": round(duration, 3),
        },
    )

    print("\n提取结果已保存:", saved_path)
    print("运行诊断已保存:", sidecar)

    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)
    print("任务状态:", result.get("status"))
    print("运行方案:", diagnostics["profile"]["label"])
    print(
        "StructureRecovery:",
        diagnostics["structure_recovery"],
    )
    print(
        "实际输入模式:",
        diagnostics.get("effective_input_mode"),
    )
    print(
        "校验模式:",
        diagnostics.get("validation_mode"),
    )
    print("提取参数数:", summary["total"])
    print("自动接受 (validated):", summary["validated"])
    print("待人工复核 (pending_review):", summary["pending_review"])
    print("拒绝 (invalid):", summary["invalid"])
    print("自动重试次数:", result.get("retry_count", 0))
    print(f"总耗时: {duration:.2f} 秒")

    print("\n自动接受参数：")
    for param in summary["auto_accepted_parameters"]:
        evidence = param.get("evidence") or {}
        print(
            f"- {param.get('name')}: "
            f"{param.get('value')} {param.get('unit')} "
            f"({param.get('value_type')}) "
            f"table={evidence.get('table_id')} "
            f"row={evidence.get('row_index')}"
        )

    if summary["pending_parameters"]:
        print("\n待人工复核参数：")
        for param in summary["pending_parameters"]:
            print(
                f"- {param.get('name')}: "
                f"{param.get('value')} {param.get('unit')} "
                f"errors={param.get('validation_errors', [])}"
            )

    if summary["invalid_parameters"]:
        print("\n被拒绝参数：")
        for param in summary["invalid_parameters"]:
            print(
                f"- {param.get('name')}: "
                f"{param.get('value')} {param.get('unit')} "
                f"errors={param.get('validation_errors', [])}"
            )

    print("\nLLM Token 使用情况：")
    for key, value in extractor.get_usage().items():
        print(f"{key}: {value}")

    print("\nAgent 执行结束。")

    return {
        "result": result,
        "run": run,
        "artifact_path": str(saved_path),
        "sidecar_path": str(sidecar),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Datasheet Copilot E2E"
    )

    parser.add_argument(
        "pdf",
        help="Datasheet PDF 文件路径",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "提取结果 JSON 输出路径"
            "（默认 data/outputs/<stem>.extraction.json）"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("d", "c"),
        default=None,
        help=(
            "运行方案；默认读取 DATASHEET_RUN_PROFILE，"
            "未设置时使用 d（Logical + P1）"
        ),
    )
    parser.add_argument(
        "--table-ids",
        default=None,
        help="逗号分隔的目标 table_id；默认由表格选择器决定",
    )

    args = parser.parse_args()

    try:
        run_e2e(
            args.pdf,
            output_json=args.output_json,
            profile=args.profile,
            table_ids=parse_table_ids(args.table_ids),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
