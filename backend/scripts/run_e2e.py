
import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from app.parsers.pdf_parser import PDFParser
from app.parsers.table_parser import TableParser

from app.parsers.table_selector import (
    TableSelector,
    SelectedTableParser,
)

from app.parsers.table_cache import TableParseCache

from app.extraction.deepseek_extractor import (
    DeepSeekExtractor
)

from app.extraction.artifact_store import (
    RecordingExtractor,
    artifact_from_state,
    file_sha256,
    save_artifact,
)

from app.agents.graph import build_graph


DEFAULT_OUTPUT_DIR = Path("data/outputs")


def default_output_path(pdf_path: str) -> Path:
    """默认提取结果保存路径。"""

    return (
        DEFAULT_OUTPUT_DIR
        / f"{Path(pdf_path).stem}.extraction.json"
    )


def run_e2e(pdf_path: str, output_json=None):
    """运行真实 Datasheet 端到端提参流程。"""

    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).is_file():
        raise FileNotFoundError(
            f"找不到 PDF 文件: {pdf_path}"
        )

    print("=" * 60)
    print("Datasheet Copilot - E2E")
    print("=" * 60)

    print(f"PDF 文件: {pdf_path}")

    # 1. 初始化工具
    pdf_parser = PDFParser()

    raw_table_parser = TableParser()

    selector = TableSelector(
        max_tables=4,
        max_chars=40000,
    )

    cache = TableParseCache.from_env()

    table_parser = SelectedTableParser(
        raw_table_parser,
        selector,
        cache=cache,
    )

    extractor = DeepSeekExtractor()

    # 在不修改 DeepSeekExtractor 的前提下，
    # 记录其完整原始结构化输出。
    recording_extractor = RecordingExtractor(
        extractor
    )

    # 证据校验模式：默认 legacy；recovered 必须显式启用。
    evidence_validation_mode = os.getenv(
        "EVIDENCE_VALIDATION_MODE", "legacy"
    ).strip().lower()

    recovered_evidence = None

    if evidence_validation_mode == "recovered":
        logical_rows_path = os.getenv(
            "RECOVERED_EVIDENCE_LOGICAL_ROWS",
            "data/outputs/TPS7A02.logical_rows.v1.json",
        )
        raw_structure_path = os.getenv(
            "RECOVERED_EVIDENCE_RAW_STRUCTURE",
            "data/outputs/TPS7A02.raw_structure.v1.json",
        )

        recovered_evidence = {
            "logical_rows": json.loads(
                Path(logical_rows_path).read_text(
                    encoding="utf-8"
                )
            ),
            "raw_structure": json.loads(
                Path(raw_structure_path).read_text(
                    encoding="utf-8"
                )
            ),
            "pdf_sha256": file_sha256(pdf_path),
            "on_error": os.getenv(
                "RECOVERED_EVIDENCE_ON_ERROR",
                "pending_review",
            ),
        }

    # 提取输入模式：默认 flat；logical 必须显式启用。
    extraction_input_mode = os.getenv(
        "EXTRACTION_INPUT_MODE", "flat"
    ).strip().lower()

    logical_extraction = None

    if extraction_input_mode == "logical":
        if recovered_evidence is not None:
            logical_rows = recovered_evidence[
                "logical_rows"
            ]
            raw_structure = recovered_evidence[
                "raw_structure"
            ]
        else:
            logical_rows_path = os.getenv(
                "RECOVERED_EVIDENCE_LOGICAL_ROWS",
                "data/outputs/TPS7A02.logical_rows.v1.json",
            )
            raw_structure_path = os.getenv(
                "RECOVERED_EVIDENCE_RAW_STRUCTURE",
                "data/outputs/TPS7A02.raw_structure.v1.json",
            )

            logical_rows = json.loads(
                Path(logical_rows_path).read_text(
                    encoding="utf-8"
                )
            )
            raw_structure = json.loads(
                Path(raw_structure_path).read_text(
                    encoding="utf-8"
                )
            )

        table_ids_env = os.getenv(
            "EXTRACTION_INPUT_TABLE_IDS", ""
        ).strip()

        logical_extraction = {
            "logical_rows": logical_rows,
            "raw_structure": raw_structure,
            "table_ids": (
                [
                    int(value)
                    for value in table_ids_env.split(",")
                    if value.strip()
                ]
                if table_ids_env
                else None
            ),
        }

    # 2. 构建 LangGraph
    graph = build_graph(
        pdf_parser=pdf_parser,
        table_parser=table_parser,
        extractor=recording_extractor,
        evidence_validation_mode=evidence_validation_mode,
        recovered_evidence=recovered_evidence,
        extraction_input_mode=extraction_input_mode,
        logical_extraction=logical_extraction,
    )

    # 3. 设置独立任务 ID
    task_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": task_id
        }
    }

    # 4. 初始化 Agent State
    initial_state = {
        "document_path": pdf_path,
        "status": "pending",
        "retry_count": 0,
        "max_retries": 1,

        # 首次真实测试暂不等待人工输入
        "enable_human_review": False,

        "errors": [],
    }

    print("\n开始执行 Agent...")
    print(f"Task ID: {task_id}")

    start_time = time.perf_counter()

    # 5. 执行整个 Agent 工作流
    result = graph.invoke(
        initial_state,
        config=config,
    )

    duration = time.perf_counter() - start_time

    # 6. 保存完整提取结果（含 evidence 与来源信息）
    if output_json is None:
        output_json = default_output_path(pdf_path)

    cache_key = None
    cache_dir = None

    if cache is not None:
        cache_key = cache.build_key(
            pdf_path,
            table_parser._config_signature(),
        )
        cache_dir = Path(cache.cache_dir).resolve()

    artifact = artifact_from_state(
        result,
        recording_extractor,
        pdf_path=pdf_path,
        pdf_sha256=file_sha256(pdf_path),
        table_cache_dir=cache_dir,
        table_cache_key=cache_key,
        model=getattr(extractor, "model", None),
    )

    saved_path = save_artifact(
        artifact,
        output_json,
    )

    print("\n提取结果已保存:", saved_path)

    # 7. 读取结果
    document_info = result.get(
        "document_info", {}
    )

    validation = result.get(
        "validation_result", {}
    )

    parameters = validation.get(
        "parameters", []
    )

    summary = validation.get(
        "validation_summary", {}
    )

    # 8. 打印摘要
    print("\n" + "=" * 60)
    print("执行结果")
    print("=" * 60)

    print(
        "任务状态:",
        result.get("status")
    )

    print(
        "PDF 页数:",
        document_info.get("page_count")
    )

    print(
        "候选表格数:",
        document_info.get("table_count")
    )

    print(
        "提取参数数:",
        len(parameters)
    )

    print(
        "验证通过:",
        summary.get("validated", 0)
    )

    print(
        "验证失败:",
        summary.get("invalid", 0)
    )

    print(
        "待人工复核:",
        summary.get("pending_review", 0)
    )

    print(
        "自动重试次数:",
        result.get("retry_count", 0)
    )

    print(
        f"总耗时: {duration:.2f} 秒"
    )

    # 9. 输出各参数
    print("\n参数提取明细：")

    for index, param in enumerate(
        parameters,
        start=1
    ):
        print("-" * 50)

        print(f"[{index}] {param['name']}")

        print(
            "数值:",
            param["value"],
            param["unit"]
        )

        print(
            "类型:",
            param["value_type"]
        )

        print(
            "测试条件:",
            param.get("test_conditions")
        )

        print(
            "校验状态:",
            param.get("validation_status")
        )

        print(
            "错误信息:",
            param.get("validation_errors", [])
        )

        print(
            "来源:",
            param.get("evidence")
        )

    # 10. 模型用量
    print("\nLLM Token 使用情况：")

    usage = extractor.get_usage()

    for key, value in usage.items():
        print(f"{key}: {value}")

    print("\nAgent 执行结束。")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Datasheet Copilot E2E"
    )

    parser.add_argument(
        "pdf",
        help="Datasheet PDF 文件路径"
    )

    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "提取结果 JSON 输出路径"
            "（默认 data/outputs/<stem>.extraction.json）"
        ),
    )

    args = parser.parse_args()

    run_e2e(
        args.pdf,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
