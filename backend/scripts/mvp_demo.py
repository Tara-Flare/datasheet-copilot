"""MVP 真实 E2E 演示脚本。

用法示例：
    python backend/scripts/mvp_demo.py \
        data/samples/b35b/onsemi_ncp170.pdf \
        --profile d --table-ids 4 \
        --output-dir data/outputs/mvp_demo/onsemi_ncp170

脚本只调用真实生产链路，不写死任何参数值，也不绕过 Validator。
输出包括运行诊断、最终校验结果、参数 CSV、原始模型输出和日志。
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (PROJECT_ROOT / "backend",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.extraction.deepseek_extractor import (  # noqa: E402
    DeepSeekExtractor,
)
from app.extraction.evidence import verify_evidence  # noqa: E402
from app.pipeline import (  # noqa: E402
    build_runtime_inputs,
    parse_table_ids,
    resolve_run_profile,
    run_pipeline,
    summarize_validation,
)
from app.validation.recovered_integration import (  # noqa: E402
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator  # noqa: E402


DEFAULT_DOCLING_ARTIFACTS = PROJECT_ROOT / "docling-models"


def configure_environment():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if (
        not os.getenv("DOCLING_ARTIFACTS_PATH")
        and DEFAULT_DOCLING_ARTIFACTS.is_dir()
    ):
        os.environ["DOCLING_ARTIFACTS_PATH"] = str(
            DEFAULT_DOCLING_ARTIFACTS
        )


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parameter_rows(parameters):
    for param in parameters:
        evidence = param.get("evidence") or {}
        yield {
            "name": param.get("name"),
            "value": param.get("value"),
            "unit": param.get("unit"),
            "value_type": param.get("value_type"),
            "test_conditions": param.get("test_conditions"),
            "table_id": evidence.get("table_id"),
            "row_index": evidence.get("row_index"),
            "page": evidence.get("page"),
            "source_text": evidence.get("source_text"),
            "validation_status": param.get(
                "validation_status"
            ),
            "validation_source": param.get(
                "validation_source"
            ),
            "detailed_recovery_status": param.get(
                "detailed_recovery_status"
            ),
            "identity_ambiguous": param.get(
                "identity_ambiguous"
            ),
            "recovered_logical_row_id": param.get(
                "recovered_logical_row_id"
            ),
            "validation_errors": "; ".join(
                param.get("validation_errors", [])
            ),
        }


def write_parameters_csv(path, parameters):
    rows = list(parameter_rows(parameters))
    fieldnames = [
        "name",
        "value",
        "unit",
        "value_type",
        "test_conditions",
        "table_id",
        "row_index",
        "page",
        "source_text",
        "validation_status",
        "validation_source",
        "detailed_recovery_status",
        "identity_ambiguous",
        "recovered_logical_row_id",
        "validation_errors",
    ]

    with Path(path).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def unit_audit(parameters, logical_rows):
    if not isinstance(logical_rows, dict):
        return []

    by_id = {
        row.get("logical_row_id"): row
        for row in logical_rows.get("logical_rows", [])
    }
    audit = []

    for param in parameters:
        row_id = param.get("recovered_logical_row_id")
        row = by_id.get(row_id)

        if row is None:
            continue

        predicted = str(param.get("unit") or "").strip()
        source = str(row.get("unit") or "").strip()

        audit.append({
            "name": param.get("name"),
            "value": param.get("value"),
            "value_type": param.get("value_type"),
            "logical_row_id": row_id,
            "source_unit": source,
            "predicted_unit": predicted,
            "unit_match": (
                predicted == source
                if source and predicted
                else None
            ),
            "row_index": (
                row.get("source_row_indices") or [None]
            )[0],
            "validation_status": param.get(
                "validation_status"
            ),
        })

    return audit


def build_summary(
    pdf_path,
    run,
    *,
    elapsed_s,
    usage,
    output_dir,
):
    diagnostics = run["diagnostics"]
    validation = run["validation"]
    parameters = validation.get("parameters", [])
    summary = run["summary"]
    profile = run["profile"]

    logical_rows = run["runtime"].get("logical_rows")
    audit = unit_audit(parameters, logical_rows)

    glyph = []
    for table in run["runtime"]["table_result"]["tables"]:
        for entry in table.get("glyph_evidence", []) or []:
            glyph.append({
                "table_id": table.get("table_id"),
                "cell_id": entry.get("cell_id"),
                "docling_text": entry.get("docling_text"),
                "pdf_text": entry.get("pdf_text"),
            })

    identity_conflicts = [
        param for param in parameters
        if "recovered_conflict:identity"
        in (param.get("validation_errors") or [])
    ]
    ambiguous_conflicts = [
        param for param in identity_conflicts
        if param.get("identity_ambiguous")
    ]
    other_identity_conflicts = [
        param for param in identity_conflicts
        if not param.get("identity_ambiguous")
    ]

    lines = [
        "# MVP E2E 演示结果",
        "",
        f"- PDF：`{pdf_path}`",
        f"- 运行方案：{profile.key.upper()} "
        f"({profile.label})",
        f"- 输入模式：`{diagnostics.get('effective_input_mode')}`",
        f"- 校验模式：`{diagnostics.get('validation_mode')}`",
        f"- Prompt 族：`{profile.prompt_family}`",
        f"- 重试次数上限：{run['state'].get('max_retries')}",
        f"- 耗时：{elapsed_s:.2f} s",
        f"- Token：prompt={usage.get('prompt_tokens')}，"
        f"completion={usage.get('completion_tokens')}，"
        f"total={usage.get('total_tokens')}",
        "",
        "## 结构恢复与输入",
        "",
        "```json",
        json.dumps(
            diagnostics.get("structure_recovery", {}),
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 状态统计",
        "",
        "| 状态 | 数量 | 含义 |",
        "| --- | ---: | --- |",
        f"| validated | {summary['validated']} | "
        "可自动接受 |",
        f"| pending_review | {summary['pending_review']} | "
        "必须人工复核，不算成功 |",
        f"| invalid | {summary['invalid']} | "
        "阻止自动接受 |",
        f"| total | {summary['total']} | 全部模型输出 |",
        "",
        "## 自动接受参数",
        "",
    ]

    if summary["auto_accepted_parameters"]:
        lines += [
            "| 参数 | 值 | 单位 | 类型 | Table | Row |",
            "| --- | ---: | --- | --- | ---: | ---: |",
        ]
        for param in summary["auto_accepted_parameters"]:
            evidence = param.get("evidence") or {}
            lines.append(
                f"| {param.get('name')} | "
                f"{param.get('value')} | "
                f"{param.get('unit')} | "
                f"{param.get('value_type')} | "
                f"{evidence.get('table_id')} | "
                f"{evidence.get('row_index')} |"
            )
    else:
        lines.append("无。")

    lines += ["", "## 待人工复核参数", ""]
    if summary["pending_parameters"]:
        for param in summary["pending_parameters"]:
            lines.append(
                f"- {param.get('name')} "
                f"{param.get('value')} {param.get('unit')}："
                f"{param.get('validation_errors', [])}"
            )
    else:
        lines.append("无。")

    lines += ["", "## 被拒绝参数", ""]
    if summary["invalid_parameters"]:
        for param in summary["invalid_parameters"]:
            lines.append(
                f"- {param.get('name')} "
                f"{param.get('value')} {param.get('unit')}："
                f"{param.get('validation_errors', [])}"
            )
    else:
        lines.append("无。")

    lines += [
        "",
        "## 单位审计（预测 vs Logical 来源）",
        "",
    ]
    if audit:
        lines += [
            "| 参数 | 值 | 类型 | 来源单位 | 预测单位 | 一致 |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
        for item in audit:
            lines.append(
                f"| {item['name']} | {item['value']} | "
                f"{item['value_type']} | "
                f"`{item['source_unit']}` | "
                f"`{item['predicted_unit']}` | "
                f"{item['unit_match']} |"
            )
    else:
        lines.append(
            "当前方案没有 recovered 来源映射，"
            "或模型输出未匹配到逻辑行。"
        )

    lines += [
        "",
        "## 字形恢复证据",
        "",
    ]
    if glyph:
        lines += [
            "| Table | Cell | Docling | PDF |",
            "| ---: | ---: | --- | --- |",
        ]
        for item in glyph:
            lines.append(
                f"| {item['table_id']} | {item['cell_id']} | "
                f"`{item['docling_text']}` | "
                f"`{item['pdf_text']}` |"
            )
    else:
        lines.append("无。")

    lines += [
        "",
        "## identity 冲突",
        "",
        f"共 {len(identity_conflicts)} 条参数因 identity 不匹配"
        "被拒绝，其中同名歧义 "
        f"{len(ambiguous_conflicts)} 条，"
        "其他来源行身份不可区分 "
        f"{len(other_identity_conflicts)} 条。",
        "",
    ]
    for param in identity_conflicts:
        flag = (
            "同名歧义"
            if param.get("identity_ambiguous")
            else "来源行身份不可区分"
        )
        lines.append(
            f"- {param.get('name')} "
            f"{param.get('value')} {param.get('unit')} "
            f"({param.get('value_type')})：{flag}"
        )

    lines += [
        "",
        "## 输出文件",
        "",
        f"- `{output_dir / 'pipeline.json'}`",
        f"- `{output_dir / 'validation.json'}`",
        f"- `{output_dir / 'parameters.csv'}`",
        f"- `{output_dir / 'raw_responses.json'}`",
        f"- `{output_dir / 'run.log'}`",
        "",
    ]

    return "\n".join(lines), {
        "unit_audit": audit,
        "glyph_evidence": glyph,
        "identity_conflict_count": len(
            identity_conflicts
        ),
    }


def _raw_extraction_from_validation(validation):
    """从在线校验结果恢复“模型原始输出”字段。

    只保留模型实际输出的字段，剥离 Validator 附加状态，
    以便用最终生产 Validator 重放，而不新增 API 调用。
    """

    allowed = {
        "name",
        "value",
        "unit",
        "value_type",
        "test_conditions",
        "evidence",
    }
    parameters = []

    for param in validation.get("parameters", []):
        item = {
            key: param.get(key)
            for key in allowed
        }
        item["evidence"] = dict(
            param.get("evidence") or {}
        )
        parameters.append(item)

    return {
        "component_model": validation.get(
            "component_model", ""
        ),
        "manufacturer": validation.get("manufacturer", ""),
        "parameters": parameters,
    }


def revalidate_demo(
    output_dir,
    pdf_path,
    *,
    profile=None,
    table_ids=None,
):
    """用最终生产 Validator 重放已保存在线输出。"""

    output_dir = Path(output_dir)
    validation_path = output_dir / "validation.json"
    pipeline_path = output_dir / "pipeline.json"

    if not validation_path.is_file():
        raise SystemExit(
            f"缺少在线输出: {validation_path}"
        )

    validation = json.loads(
        validation_path.read_text(encoding="utf-8")
    )
    saved_pipeline = {}
    if pipeline_path.is_file():
        saved_pipeline = json.loads(
            pipeline_path.read_text(encoding="utf-8")
        )

    diagnostics = saved_pipeline.get("diagnostics", {})
    saved_profile = (
        diagnostics.get("profile", {}).get("key")
    )
    resolved = resolve_run_profile(
        profile or saved_profile
    )
    saved_table_ids = (
        diagnostics.get("table_selection", {}).get(
            "requested_table_ids"
        )
    )

    runtime = build_runtime_inputs(
        pdf_path,
        profile=resolved,
        table_ids=(
            parse_table_ids(table_ids)
            if table_ids is not None
            else saved_table_ids
        ),
    )

    extraction = _raw_extraction_from_validation(validation)
    legacy = ParameterValidator().validate(
        verify_evidence(
            extraction,
            runtime["table_result"]["tables"],
        )
    )

    if resolved.evidence_validation_mode == "recovered":
        final = build_recovered_validation_result(
            extraction,
            legacy,
            runtime["recovered_evidence"],
        )
    else:
        final = legacy

    before = {
        item.get("name"): item.get("validation_status")
        for item in validation.get("parameters", [])
    }
    changes = []
    for index, item in enumerate(
        final.get("parameters", [])
    ):
        before_status = before.get(item.get("name"))
        after_status = item.get("validation_status")
        if before_status != after_status:
            changes.append({
                "index": index,
                "name": item.get("name"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "value_type": item.get("value_type"),
                "before_status": before_status,
                "after_status": after_status,
                "after_errors": item.get(
                    "validation_errors", []
                ),
            })

    write_json(
        output_dir / "validation_revalidated.json",
        final,
    )
    write_json(
        output_dir / "revalidation_diff.json",
        {
            "schema": "mvp_demo_revalidation/v1",
            "note": (
                "使用已保存的在线模型输出重放最终生产 "
                "Validator；不新增 API 调用。"
            ),
            "before_counts": _status_counts(validation),
            "after_counts": _status_counts(final),
            "changed_count": len(changes),
            "changes": changes,
        },
    )
    write_parameters_csv(
        output_dir / "parameters.csv",
        final.get("parameters", []),
    )

    run = {
        "state": {
            "max_retries": 1,
        },
        "runtime": runtime,
        "diagnostics": runtime["diagnostics"],
        "validation": final,
        "summary": summarize_validation(final),
        "profile": resolved,
    }
    summary_text, _ = build_summary(
        pdf_path,
        run,
        elapsed_s=saved_pipeline.get("elapsed_s", 0.0),
        usage=saved_pipeline.get("usage", {}),
        output_dir=output_dir,
    )
    summary_text += (
        "\n> 本报告使用最终生产 Validator 对已保存在线输出"
        "重放后生成；原始 `validation.json` 保留未覆盖，"
        "最终结果见 `validation_revalidated.json`。\n"
    )
    (output_dir / "summary.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "before_counts": _status_counts(validation),
            "after_counts": _status_counts(final),
            "changed_count": len(changes),
            "changes": changes,
        },
        ensure_ascii=False,
        indent=2,
    ))

    return {
        "validation": final,
        "runtime": runtime,
        "changes": changes,
    }


def _status_counts(validation):
    counts = {}

    for item in validation.get("parameters", []):
        status = item.get("validation_status")
        counts[status] = counts.get(status, 0) + 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Datasheet Copilot MVP 真实 E2E 演示"
    )
    parser.add_argument("pdf", help="Datasheet PDF 路径")
    parser.add_argument(
        "--profile",
        choices=("d", "c"),
        default=None,
        help="运行方案；默认 d",
    )
    parser.add_argument(
        "--table-ids",
        default=None,
        help="逗号分隔的 table_id；不指定时由选择器决定",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="演示输出目录",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="自动纠错重试次数",
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help=(
            "不调用 LLM，使用已保存在线输出重放最终 "
            "生产 Validator"
        ),
    )

    args = parser.parse_args()

    configure_environment()

    profile = resolve_run_profile(args.profile)
    pdf_path = Path(args.pdf).resolve()

    if not args.revalidate and not pdf_path.is_file():
        parser.error(
            f"找不到 PDF 文件: {pdf_path}；"
            "请自备 Datasheet，公开仓库不包含 PDF 样本"
        )

    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT
        / "data/outputs/mvp_demo"
        / pdf_path.stem
    )

    if args.revalidate:
        revalidate_demo(
            output_dir,
            pdf_path,
            profile=args.profile,
            table_ids=args.table_ids,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("mvp_demo")
    logger.info(
        "开始演示: pdf=%s profile=%s table_ids=%s",
        pdf_path,
        profile.key,
        args.table_ids,
    )

    try:
        extractor = DeepSeekExtractor(
            family_coverage=profile.family_coverage,
        )
    except ValueError as exc:
        logger.error("演示配置错误: %s", exc)
        parser.error(str(exc))

    started = time.perf_counter()

    try:
        run = run_pipeline(
            pdf_path,
            profile=profile,
            extractor=extractor,
            table_ids=parse_table_ids(args.table_ids),
            max_retries=args.max_retries,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("演示失败: %s", exc)
        parser.error(str(exc))

    elapsed_s = time.perf_counter() - started
    usage = extractor.get_usage()

    write_json(
        output_dir / "pipeline.json",
        {
            "schema": "mvp_demo_pipeline/v1",
            "pdf_path": str(pdf_path),
            "profile": profile.to_dict(),
            "diagnostics": run["diagnostics"],
            "state_status": run["state"].get("status"),
            "elapsed_s": round(elapsed_s, 3),
            "usage": usage,
        },
    )
    write_json(
        output_dir / "validation.json",
        run["validation"],
    )
    write_json(
        output_dir / "raw_responses.json",
        extractor.raw_responses,
    )
    write_parameters_csv(
        output_dir / "parameters.csv",
        run["validation"].get("parameters", []),
    )

    summary_text, extras = build_summary(
        pdf_path,
        run,
        elapsed_s=elapsed_s,
        usage=usage,
        output_dir=output_dir,
    )
    (output_dir / "summary.md").write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text)

    logger.info(
        "演示完成: validated=%s pending=%s invalid=%s",
        run["summary"]["validated"],
        run["summary"]["pending_review"],
        run["summary"]["invalid"],
    )

    return {
        "run": run,
        "extras": extras,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    main()
