"""Datasheet Copilot 离线重校验脚本。

用法示例：

    PYTHONPATH=backend .venv/bin/python \
        backend/scripts/run_offline_validation.py \
        --extraction data/outputs/TPS7A02.extraction.json \
        --output data/outputs/TPS7A02.validation.json

    # 与旧版本校验结果对比
    PYTHONPATH=backend .venv/bin/python \
        backend/scripts/run_offline_validation.py \
        --extraction data/outputs/TPS7A02.extraction.json \
        --compare data/outputs/TPS7A02.validation.old.json

本脚本只读取提取结果 JSON 与 Docling 表格缓存，
不调用 DeepSeek API，也不重新解析 PDF。
"""

import argparse
import json
import sys
from pathlib import Path

from app.extraction.artifact_store import (
    ArtifactError,
    load_artifact,
)

from app.validation.offline import (
    OfflineValidationError,
    build_validation_report,
    compare_reports,
    ensure_tables_available,
    load_cached_tables,
    load_validation_report,
    save_validation_report,
    select_extraction,
)


def resolve_cache_path(
    artifact,
    explicit_path=None,
) -> Path:
    """定位 Docling 表格缓存文件。"""

    if explicit_path:
        return Path(explicit_path)

    source = artifact.get("source") or {}

    cache_dir = source.get("table_cache_dir")
    cache_key = source.get("table_cache_key")

    if cache_dir and cache_key:
        return Path(cache_dir) / cache_key

    raise OfflineValidationError(
        "无法定位 Docling 表格缓存: 请通过 "
        "--table-cache 指定缓存 JSON 文件"
    )


def run(
    extraction_path,
    *,
    table_cache=None,
    output=None,
    validator_version=None,
    source="extraction",
    compare=None,
) -> dict:
    """执行一次离线重校验并保存报告。"""

    artifact = load_artifact(extraction_path)

    cache_path = resolve_cache_path(
        artifact,
        table_cache,
    )

    tables = load_cached_tables(cache_path)

    extraction = select_extraction(
        artifact,
        source,
    )

    ensure_tables_available(
        tables,
        extraction,
    )

    report = build_validation_report(
        extraction,
        tables,
        validator_version=validator_version,
        extraction_source=source,
        artifact_path=extraction_path,
        extraction_metadata=artifact.get("source"),
    )

    if output is None:
        output = Path(extraction_path).with_suffix(
            ".validation.json"
        )

    output = save_validation_report(
        report,
        output,
    )

    summary = report["summary"]
    statistics = report["error_statistics"]

    print("离线校验完成")
    print("提取结果:", extraction_path)
    print("表格缓存:", cache_path)
    print("校验结果:", output)
    print(
        "参数总数: %s, 通过: %s, 失败: %s, 待复核: %s"
        % (
            summary.get("total", 0),
            summary.get("validated", 0),
            summary.get("invalid", 0),
            summary.get("pending_review", 0),
        )
    )
    print(
        "错误总数: %s, 含错误参数: %s"
        % (
            statistics["total_errors"],
            statistics["parameters_with_errors"],
        )
    )

    for category, count in sorted(
        statistics["by_category"].items()
    ):
        print(f"  - {category}: {count}")

    if compare:
        previous = load_validation_report(compare)

        diff = compare_reports(previous, report)

        diff_path = Path(output).with_suffix(
            ".diff.json"
        )

        diff_path.write_text(
            json.dumps(
                diff,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("版本对比")
        print(
            "  旧版本:",
            diff["old_validator_version"],
        )
        print(
            "  新版本:",
            diff["new_validator_version"],
        )
        print("  改善:", diff["improved"])
        print("  回退:", diff["regressed"])
        print(
            "  变化总数:",
            diff["changed_count"],
        )
        print("差异明细:", diff_path)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Datasheet Copilot 离线重校验"
            "（不调用 DeepSeek API）"
        )
    )

    parser.add_argument(
        "--extraction",
        required=True,
        help="保存的提取结果 JSON",
    )

    parser.add_argument(
        "--table-cache",
        default=None,
        help=(
            "Docling 表格缓存 JSON；"
            "默认读取提取结果中记录的位置"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="校验结果输出路径",
    )

    parser.add_argument(
        "--validator-version",
        default=None,
        help="自定义 Validator 版本标签",
    )

    parser.add_argument(
        "--source",
        choices=[
            "extraction",
            "final_extraction",
        ],
        default="extraction",
        help="选择校验哪一份提取结果",
    )

    parser.add_argument(
        "--compare",
        default=None,
        help="与已有校验结果 JSON 对比",
    )

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        run(
            args.extraction,
            table_cache=args.table_cache,
            output=args.output,
            validator_version=args.validator_version,
            source=args.source,
            compare=args.compare,
        )

    except (ArtifactError, OfflineValidationError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
