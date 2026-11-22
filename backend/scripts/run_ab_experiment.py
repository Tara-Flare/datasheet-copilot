"""阶段 B3.2b：TPS7A02 table_id=9 的小规模 Flat / Logical A/B 实验。

安全约束：
- 默认只做预检/准备，不调用真实 LLM；
- 必须显式 --execute 才会访问 DeepSeek；
- 只处理目标 table_id，不运行完整 PDF E2E；
- 不做自动 Retry；
- 使用全新输出目录，已存在则拒绝写入；
- 不修改 data/outputs 下已有文件。
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.extraction.deepseek_extractor import (
    DeepSeekExtractor,
)
from app.extraction.evidence import verify_evidence
from app.extraction.input_mode import (
    build_extraction_input,
)
from app.extraction.logical_table_adapter import (
    adapt_logical_rows,
)
from app.evaluation.ab_scoring import (
    score_extraction,
)
from app.validation.recovered_evidence import (
    RecoveredEvidenceValidator,
)
from app.validation.recovered_integration import (
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator


DEFAULT_LOGICAL_ROWS = str(
    PROJECT_ROOT
    / "data/outputs/TPS7A02.logical_rows.v2.json"
)
DEFAULT_RAW_STRUCTURE = str(
    PROJECT_ROOT
    / "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_GOLD = str(
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)
DEFAULT_CACHE_DIR = str(
    PROJECT_ROOT / "data/cache/docling"
)
DEFAULT_OUTPUT_ROOT = str(
    PROJECT_ROOT / "data/outputs/experiments/B3.2b"
)
DEFAULT_TABLE_ID = 9


class ExperimentError(ValueError):
    """A/B 实验预检或写入失败。"""


def load_json(path):
    path = Path(path)

    if not path.is_file():
        raise ExperimentError(
            f"文件不存在: {path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_exclusive(path, data):
    """只创建新文件，绝不覆盖已有文件。"""

    path = Path(path)

    if path.exists():
        raise ExperimentError(
            f"拒绝覆盖已有文件: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    return path


def write_text_exclusive(path, text):
    """只创建新文本文件，绝不覆盖已有文件。"""

    path = Path(path)

    if path.exists():
        raise ExperimentError(
            f"拒绝覆盖已有文件: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    return path


def find_flat_table(cache_dir, table_id):
    """从 Docling 缓存中只加载目标表格。"""

    cache_dir = Path(cache_dir)

    if not cache_dir.is_dir():
        raise ExperimentError(
            f"表格缓存目录不存在: {cache_dir}"
        )

    matches = []

    for path in sorted(cache_dir.glob("*.json")):
        data = load_json(path)

        for table in data.get("tables", []):
            if table.get("table_id") == table_id:
                matches.append((path, table))

    if not matches:
        raise ExperimentError(
            f"缓存中找不到 table_id={table_id}"
        )

    if len(matches) > 1:
        raise ExperimentError(
            f"缓存中 table_id={table_id} 不唯一: "
            f"{[str(path) for path, _ in matches]}"
        )

    path, table = matches[0]

    return copy.deepcopy(table), path


def build_gold_standard(
    logical_rows,
    raw_structure,
    table_id,
):
    """由已确认 Logical 行生成人工核验标准答案。"""

    adapted = adapt_logical_rows(
        logical_rows,
        raw_structure=raw_structure,
        table_ids=[table_id],
    )

    table = adapted["tables"][0]
    expected = []

    for record in table["records"]:
        for value_type in ("min", "typ", "max"):
            value = str(
                record.get(value_type) or ""
            ).strip()

            if not value:
                continue

            expected.append({
                "logical_row_id": record[
                    "logical_row_id"
                ],
                "table_id": table_id,
                "page": record.get("page"),
                "parameter_symbol": record.get(
                    "parameter_symbol"
                ),
                "parameter_name": record.get(
                    "parameter_name"
                ),
                "value": value,
                "value_type": value_type,
                "unit": record.get("unit"),
                "test_conditions": record.get(
                    "test_conditions"
                ),
                "source_row_indices": record.get(
                    "source_row_indices"
                ),
                "source_cell_ids": record.get(
                    "source_cell_ids"
                ),
                "needs_review": bool(
                    record.get("needs_review")
                ),
            })

    excluded = []

    for item in table.get("excluded_rows", []):
        excluded.append({
            "logical_row_id": item.get(
                "logical_row_id"
            ),
            "source_row_indices": item.get(
                "source_row_indices"
            ),
            "source_cell_ids": item.get(
                "source_cell_ids"
            ),
            "needs_review": item.get("needs_review"),
            "reason": item.get("reason"),
        })

    return {
        "schema": "ab_gold_standard/v1",
        "table_id": table_id,
        "source": {
            "pdf_sha256": (
                raw_structure.get("source") or {}
            ).get("pdf_sha256"),
            "logical_rows_schema": logical_rows.get(
                "schema"
            ),
            "logical_rows_version": logical_rows.get(
                "logical_rows_version"
            ),
        },
        "trusted_logical_row_count": len(
            table["records"]
        ),
        "expected_parameter_count": len(expected),
        "expected_parameters": expected,
        "excluded_needs_review": excluded,
        "notes": [
            "标准答案来自 B3.1 已确认的 Logical 行，"
            "不是 LLM 提取结果。",
            "needs_review=true 的行不进入可信输入，"
            "单独列在 excluded_needs_review。",
            "人工核验时应逐条比对 value_type、value、"
            "unit、test_conditions 和来源行。",
        ],
    }


def extractor_config(extractor):
    return {
        "model": getattr(extractor, "model", None),
        "input_mode": getattr(
            extractor, "input_mode", None
        ),
        "max_tokens": getattr(
            extractor, "max_tokens", None
        ),
        "rows_per_chunk": getattr(
            extractor, "rows_per_chunk", None
        ),
        "max_split_depth": getattr(
            extractor, "max_split_depth", None
        ),
        "temperature": getattr(
            extractor,
            "temperature",
            "provider_default",
        ),
    }


def run_arm(
    name,
    input_tables,
    flat_table,
    logical_rows,
    raw_structure,
    pdf_sha256,
    input_mode,
    gold,
):
    """执行单臂首次提取，不触发 Retry。"""

    extractor = DeepSeekExtractor(
        input_mode=input_mode
    )

    started = time.perf_counter()
    extraction = extractor.extract(input_tables)
    elapsed = time.perf_counter() - started

    usage = extractor.get_usage()
    raw_responses = copy.deepcopy(
        getattr(extractor, "raw_responses", [])
    )

    legacy = ParameterValidator().validate(
        verify_evidence(extraction, [flat_table])
    )

    recovered = build_recovered_validation_result(
        extraction,
        legacy,
        {
            "logical_rows": logical_rows,
            "raw_structure": raw_structure,
            "pdf_sha256": pdf_sha256,
            "on_error": "pending_review",
        },
    )

    raw_recovered = RecoveredEvidenceValidator(
        logical_rows,
        raw_structure=raw_structure,
    ).validate(extraction)

    evaluation = score_extraction(
        gold,
        extraction,
        raw_recovered,
    )

    return {
        "arm": name,
        "input_mode": input_mode,
        "extractor_config": extractor_config(
            extractor
        ),
        "metrics": {
            "elapsed_seconds": round(elapsed, 4),
            "request_count": len(raw_responses),
            "prompt_tokens": usage.get(
                "prompt_tokens", 0
            ),
            "completion_tokens": usage.get(
                "completion_tokens", 0
            ),
            "total_tokens": usage.get(
                "total_tokens", 0
            ),
            "extracted_parameter_count": len(
                extraction.get("parameters", [])
            ),
        },
        "evaluation": evaluation,
        "extraction": extraction,
        "legacy_validation": legacy,
        "recovered_validation": recovered,
        "recovered_validation_raw": raw_recovered,
        "raw_llm_responses": raw_responses,
    }


def build_plan(args):
    logical_rows = load_json(args.logical_rows)
    raw_structure = load_json(args.raw_structure)
    flat_table, table_source = find_flat_table(
        args.table_cache_dir,
        args.table_id,
    )

    logical_prepared = build_extraction_input(
        [flat_table],
        mode="logical",
        logical_rows=logical_rows,
        raw_structure=raw_structure,
        table_ids=[args.table_id],
    )

    if logical_prepared["effective_mode"] != "logical":
        raise ExperimentError(
            "Logical 输入未覆盖目标表格，拒绝静默回退 Flat"
        )

    logical_table = logical_prepared["tables"][0]

    # 两个 arm 必须使用完全相同的 Gold v2 评估文件。
    gold = load_json(args.gold)
    gold_sha256 = sha256_file(args.gold)

    if gold.get("table_id") != args.table_id:
        raise ExperimentError(
            "Gold table_id 与实验目标不一致"
        )

    if gold.get("expected_parameter_count") != len(
        gold.get("expected_parameters", [])
    ):
        raise ExperimentError(
            "Gold expected_parameter_count 不一致"
        )

    if gold.get("expected_parameter_count") != 51:
        raise ExperimentError(
            "Gold v2 评估文件不是 51 条: "
            f"{gold.get('expected_parameter_count')}"
        )

    if not gold.get("table_default_conditions"):
        raise ExperimentError(
            "Gold v2 缺少表级默认条件"
        )

    if not all(
        str(number) in gold.get("footnotes", {})
        for number in (1, 2, 3, 4)
    ):
        raise ExperimentError(
            "Gold v2 缺少脚注 (1)～(4)"
        )

    pdf_sha256 = (
        raw_structure.get("source") or {}
    ).get("pdf_sha256")

    plan = {
        "schema": "ab_experiment_plan/v1",
        "stage": "B3.2b",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_call": bool(args.execute),
        "target": {
            "table_id": args.table_id,
            "flat_table_source": str(table_source),
            "flat_row_count": len(
                flat_table.get("rows", [])
            ),
            "logical_trusted_row_count": len(
                logical_table.get("records", [])
            ),
            "excluded_needs_review_ids": [
                item["logical_row_id"]
                for item in logical_prepared[
                    "tables"
                ][0].get("excluded_rows", [])
            ],
            "logical_rows_revision": logical_rows.get(
                "logical_rows_revision", "v1"
            ),
            "pdf_sha256": pdf_sha256,
        },
        "gold": {
            "path": str(args.gold),
            "sha256": gold_sha256,
            "schema": gold.get("schema"),
            "expected_parameter_count": gold.get(
                "expected_parameter_count"
            ),
            "table_default_conditions_present": bool(
                gold.get("table_default_conditions")
            ),
            "footnotes_present": sorted(
                gold.get("footnotes", {})
            ),
        },
        "arms": [
            {
                "name": "flat",
                "extraction_input_mode": "flat",
                "table_ids": [args.table_id],
                "row_count": len(
                    flat_table.get("rows", [])
                ),
            },
            {
                "name": "logical",
                "extraction_input_mode": "logical",
                "table_ids": [args.table_id],
                "row_count": len(
                    logical_table.get("records", [])
                ),
                "excluded_needs_review_ids": [
                    item["logical_row_id"]
                    for item in logical_table.get(
                        "excluded_rows", []
                    )
                ],
            },
        ],
        "controls": {
            "same_model_and_parameters": True,
            "retry_enabled": False,
            "full_pdf_e2e": False,
            "only_target_table": True,
            "overwrite_existing_outputs": False,
            "validation": (
                "simultaneously record legacy and recovered; "
                "recovered is authoritative for covered table"
            ),
        },
        "output_policy": {
            "output_dir": args.output_dir,
            "refuse_if_exists": True,
        },
    }

    return {
        "plan": plan,
        "gold": gold,
        "flat_table": flat_table,
        "logical_table": logical_table,
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "pdf_sha256": pdf_sha256,
    }


def prepare_outputs(args, prepared):
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise ExperimentError(
            f"拒绝使用已存在的实验目录: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    plan_path = write_json_exclusive(
        output_dir / "plan.json",
        prepared["plan"],
    )
    gold_path = write_json_exclusive(
        output_dir / "gold_standard.used.json",
        prepared["gold"],
    )

    return output_dir, plan_path, gold_path


def build_comparison(flat_arm, logical_arm):
    flat_eval = flat_arm["evaluation"]
    logical_eval = logical_arm["evaluation"]

    metric_keys = [
        "precision",
        "recall",
        "f1",
        "key_precision",
        "key_recall",
        "key_f1",
        "matched_keys",
        "strict_matched",
        "missing_count",
        "false_positive_count",
        "duplicate_count",
    ]

    deltas = {}

    for key in metric_keys:
        flat_value = flat_eval.get(key)
        logical_value = logical_eval.get(key)

        if isinstance(flat_value, (int, float)) and isinstance(
            logical_value, (int, float)
        ):
            deltas[key] = round(
                logical_value - flat_value, 4
            )
        else:
            deltas[key] = None

    flat_metrics = flat_arm["metrics"]
    logical_metrics = logical_arm["metrics"]

    for key in (
        "total_tokens",
        "request_count",
        "elapsed_seconds",
    ):
        deltas[key] = round(
            logical_metrics.get(key, 0)
            - flat_metrics.get(key, 0),
            4,
        )

    return {
        "flat": {
            **{
                key: flat_eval.get(key)
                for key in metric_keys
            },
            **flat_metrics,
            "field_accuracy": flat_eval.get(
                "field_accuracy"
            ),
            "evidence_source": flat_eval.get(
                "evidence_source"
            ),
        },
        "logical": {
            **{
                key: logical_eval.get(key)
                for key in metric_keys
            },
            **logical_metrics,
            "field_accuracy": logical_eval.get(
                "field_accuracy"
            ),
            "evidence_source": logical_eval.get(
                "evidence_source"
            ),
        },
        "delta_logical_minus_flat": deltas,
    }


def build_prompt_recommendations(flat_arm, logical_arm):
    flat_eval = flat_arm["evaluation"]
    logical_eval = logical_arm["evaluation"]
    recommendations = []

    flat_recall = flat_eval.get("recall") or 0
    logical_recall = logical_eval.get("recall") or 0

    if logical_recall > flat_recall:
        recommendations.append(
            "Logical v2 输入提升了严格 Recall；"
            "下一阶段应保留 Logical Rows 作为主输入，"
            "Flat 仅作为兼容基线。"
        )
    elif logical_recall < flat_recall:
        recommendations.append(
            "Logical v2 严格 Recall 低于 Flat，"
            "需要检查 Logical 输入列顺序、条件文本和"
            "逻辑行粒度的 Prompt 表达，暂不应切换主输入。"
        )
    else:
        recommendations.append(
            "Flat 与 Logical 严格 Recall 相同；"
            "优先分析差异样本，而不是直接切换输入模式。"
        )

    for arm in (flat_arm, logical_arm):
        name = arm["arm"]
        evaluation = arm["evaluation"]
        fields = evaluation.get("field_accuracy") or {}

        if fields.get("value") is not None and fields[
            "value"
        ] < 0.9:
            recommendations.append(
                f"{name}: 数值准确率低于 0.9；"
                "Prompt 需再次强调 Min/Typ/Max 列与数值列对应关系。"
            )

        if fields.get("unit") is not None and fields[
            "unit"
        ] < 0.9:
            recommendations.append(
                f"{name}: 单位准确率低于 0.9；"
                "Prompt 需强调从 UNIT 列复制，不得跨行推断。"
            )

        if fields.get("test_conditions") is not None and fields[
            "test_conditions"
        ] < 0.9:
            recommendations.append(
                f"{name}: 测试条件准确率低于 0.9；"
                "Prompt 需要求完整复制该逻辑行的行级测试条件，"
                "不得把表级默认条件写成行级覆盖。"
            )

        if fields.get("identity") is not None and fields[
            "identity"
        ] < 0.9:
            recommendations.append(
                f"{name}: 参数身份准确率低于 0.9；"
                "下一阶段应补充标准参数名别名映射，"
                "并检查模型是否把 Ground Current 与 "
                "Ground Current vs Load Current 混用。"
            )

        if evaluation.get(
            "false_positive_count", 0
        ) > 0:
            recommendations.append(
                f"{name}: 存在 "
                f"{evaluation['false_positive_count']} 条 "
                "false positive；Prompt 应强化"
                "“无法确定则跳过”和禁止重复输出同一列。"
            )

        evidence = evaluation.get(
            "evidence_source", {}
        )
        if (
            evidence.get("accuracy_matched") is not None
            and evidence["accuracy_matched"] < 0.9
        ):
            recommendations.append(
                f"{name}: Evidence 来源正确率低于 0.9；"
                "Prompt 应明确复制 table_id / row_index / "
                "page，Logical 模式应复制 logical_row_id。"
            )

    uvlo_flat = flat_eval.get("special", {}).get(
        "uvlo", {}
    )
    uvlo_logical = logical_eval.get(
        "special", {}
    ).get("uvlo", {})

    if (
        uvlo_logical.get("strict_recall") or 0
    ) > (
        uvlo_flat.get("strict_recall") or 0
    ):
        recommendations.append(
            "UVLO 专项应使用 Logical v2；Flat 的合并行"
            "无法稳定区分 VIN falling 与 VIN hysteresis。"
        )

    return recommendations


def _arm_detail_map(arm):
    return {
        (
            item["logical_row_id"],
            item["value_type"],
        ): item
        for item in arm["evaluation"]["gold_details"]
    }


def _detail_cell(item):
    if item is None or item["match_status"] == "missing":
        return "MISSING"

    if item["strict_match"]:
        return "OK"

    failed = [
        field
        for field in (
            "identity",
            "value",
            "unit",
            "value_type",
            "test_conditions",
        )
        if not item["fields"][field]
    ]
    return "PARTIAL: " + ",".join(failed)


def _predicted_cell(item):
    if item is None or item["match_status"] == "missing":
        return "—"

    predicted = item["predicted"]
    value = f"{predicted.get('value')} {predicted.get('unit')}"

    return (
        f"{value} [{predicted.get('value_type')}] "
        f"{predicted.get('name')}"
    )


def _evidence_cell(item):
    if item is None or item["match_status"] == "missing":
        return "—"

    check = item["evidence_check"] or {}
    evidence = item["predicted"].get("evidence") or {}

    return (
        f"{'OK' if check.get('correct') else 'FAIL'}"
        f" ({check.get('method')}) "
        f"id={evidence.get('logical_row_id')} "
        f"row={evidence.get('row_index')}"
    )


def build_report_markdown(result):
    flat = result["arms"][0]
    logical = result["arms"][1]
    comparison = result["comparison"]
    flat_details = _arm_detail_map(flat)
    logical_details = _arm_detail_map(logical)
    gold_records = result["gold_standard"][
        "expected_parameters"
    ]
    lines = [
        "# TPS7A02 table_id=9 · Flat vs Logical v2 A/B 实验报告",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        "## 实验控制",
        "",
        f"- 模型：`{flat['extractor_config']['model']}`",
        f"- temperature：`{flat['extractor_config'].get('temperature')}`",
        f"- max_tokens：`{flat['extractor_config']['max_tokens']}`",
        f"- rows_per_chunk：`{flat['extractor_config']['rows_per_chunk']}`",
        f"- max_split_depth：`{flat['extractor_config']['max_split_depth']}`",
        f"- Gold：`{result['plan']['gold']['path']}`",
        f"- Gold SHA256：`{result['plan']['gold']['sha256']}`",
        f"- Gold 记录：{result['gold_standard']['expected_parameter_count']}",
        "- Retry：未启用",
        "- 完整 PDF E2E：未执行",
        "",
        "## 质量指标",
        "",
        "| 指标 | Flat | Logical v2 | Δ (Logical-Flat) |",
        "|---|---:|---:|---:|",
    ]

    metric_rows = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Key Precision", "key_precision"),
        ("Key Recall", "key_recall"),
        ("Key F1", "key_f1"),
        ("严格匹配", "strict_matched"),
        ("键级匹配", "matched_keys"),
        ("遗漏", "missing_count"),
        ("False Positive", "false_positive_count"),
        ("重复输出", "duplicate_count"),
        ("提取参数数", "extracted_parameter_count"),
        ("请求次数", "request_count"),
        ("Prompt Tokens", "prompt_tokens"),
        ("Completion Tokens", "completion_tokens"),
        ("Total Tokens", "total_tokens"),
        ("耗时（秒）", "elapsed_seconds"),
    ]

    for label, key in metric_rows:
        flat_value = comparison["flat"].get(key)
        logical_value = comparison["logical"].get(key)
        delta = comparison[
            "delta_logical_minus_flat"
        ].get(key)
        lines.append(
            f"| {label} | {flat_value} | "
            f"{logical_value} | {delta} |"
        )

    lines += [
        "",
        "## 字段准确率（按匹配键）",
        "",
        "| 字段 | Flat | Logical v2 |",
        "|---|---:|---:|",
    ]

    for key in (
        "identity",
        "min_typ_max",
        "value",
        "unit",
        "test_conditions",
        "test_conditions_exact",
    ):
        lines.append(
            f"| {key} | "
            f"{comparison['flat']['field_accuracy'].get(key)} | "
            f"{comparison['logical']['field_accuracy'].get(key)} |"
        )

    lines += [
        "",
        "## Evidence 来源",
        "",
        "| 指标 | Flat | Logical v2 |",
        "|---|---:|---:|",
    ]

    for key in (
        "correct",
        "checked_matched",
        "accuracy_matched",
        "accuracy_all_predictions",
        "logical_row_id_present",
        "logical_row_id_presence_rate",
    ):
        lines.append(
            f"| {key} | "
            f"{comparison['flat']['evidence_source'].get(key)} | "
            f"{comparison['logical']['evidence_source'].get(key)} |"
        )

    lines += [
        "",
        "## 专项分析",
        "",
        "| 分组 | Gold 数 | Flat 严格匹配 | Logical 严格匹配 | Flat Recall | Logical Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for group_key, label in (
        ("ground_current", "Ground Current"),
        ("vdo", "VDO / Dropout"),
        ("uvlo", "UVLO"),
    ):
        flat_special = flat["evaluation"]["special"][
            group_key
        ]
        logical_special = logical["evaluation"][
            "special"
        ][group_key]
        lines.append(
            f"| {label} | {flat_special['gold_records']} | "
            f"{flat_special['strict_matched']} | "
            f"{logical_special['strict_matched']} | "
            f"{flat_special['strict_recall']} | "
            f"{logical_special['strict_recall']} |"
        )

    lines += [
        "",
        "## 51 条 Gold 逐参数匹配明细",
        "",
        "| logical_row_id | 类型 | Gold 值 | 单位 | Flat | Logical | Flat 预测 | Logical 预测 | Flat Evidence | Logical Evidence |",
        "|---|---|---:|---|---|---|---|---|---|---|",
    ]

    for gold in gold_records:
        key = (
            gold["logical_row_id"],
            gold["value_type"],
        )
        flat_item = flat_details.get(key)
        logical_item = logical_details.get(key)

        lines.append(
            f"| {gold['logical_row_id']} | "
            f"{gold['value_type']} | "
            f"{gold['value']} | {gold['unit']} | "
            f"{_detail_cell(flat_item)} | "
            f"{_detail_cell(logical_item)} | "
            f"{_predicted_cell(flat_item)} | "
            f"{_predicted_cell(logical_item)} | "
            f"{_evidence_cell(flat_item)} | "
            f"{_evidence_cell(logical_item)} |"
        )

    lines += [
        "",
        "## 错误与遗漏来源",
        "",
    ]

    for arm in (flat, logical):
        name = arm["arm"]
        evaluation = arm["evaluation"]
        lines += [
            f"### {name}",
            "",
            f"- 遗漏 {len(evaluation['missing'])} 条",
            f"- False positive {len(evaluation['false_positives'])} 条",
            f"- 字段不匹配 {len(evaluation['field_mismatches'])} 条",
            "",
        ]

        if evaluation["missing"]:
            lines.append("遗漏：")
            for item in evaluation["missing"]:
                lines.append(
                    f"- {item['logical_row_id']} "
                    f"{item['value_type']} "
                    f"{item['gold_value']} {item['unit']} "
                    f"| {item['test_conditions']}"
                )
            lines.append("")

        if evaluation["field_mismatches"]:
            lines.append("字段不匹配：")
            for item in evaluation["field_mismatches"][:30]:
                lines.append(
                    f"- {item['logical_row_id']} "
                    f"{item['value_type']} "
                    f"fields={item['fields']} "
                    f"pred={item['predicted']}"
                )
            lines.append("")

    lines += [
        "## Prompt 建议",
        "",
    ]

    for index, recommendation in enumerate(
        result["prompt_recommendations"], start=1
    ):
        lines.append(
            f"{index}. {recommendation}"
        )

    lines += [
        "",
        "## 原始响应",
        "",
        "两组完整原始响应保存在 `ab_result.json` 的 "
        "`arms[*].raw_llm_responses`，"
        "包括 content、finish_reason、Token 和错误信息。",
        "",
        "## 指标可靠性",
        "",
        "- 参数匹配以 logical_row_id + value_type 为主键。",
        "- Flat 缺少 logical_row_id 时，优先使用 recovered "
        "selected row；仍无法唯一映射的预测标记为 unmapped，"
        "不做猜测。",
        "- 表级默认条件和脚注不要求模型逐字输出；"
        "严格匹配只比较行级 test_conditions。",
        "- 如果某字段分母为 0，指标为 null，不虚构。",
        "",
    ]

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "TPS7A02 table_id=9 Flat/Logical A/B "
            "预检与最小实验"
        )
    )
    parser.add_argument(
        "--logical-rows",
        default=DEFAULT_LOGICAL_ROWS,
    )
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD,
        help="两个 arm 共用的 Gold v2 评估文件",
    )
    parser.add_argument(
        "--raw-structure",
        default=DEFAULT_RAW_STRUCTURE,
    )
    parser.add_argument(
        "--table-cache-dir",
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--table-id",
        type=int,
        default=DEFAULT_TABLE_ID,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="显式允许调用真实 DeepSeek API",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只写 plan 和 gold standard，不调用 API",
    )

    args = parser.parse_args(argv)

    if args.output_dir is None:
        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = str(
            Path(DEFAULT_OUTPUT_ROOT) / stamp
        )

    if args.execute and args.prepare_only:
        raise ExperimentError(
            "--execute 与 --prepare-only 不能同时使用"
        )

    if args.execute and not os.getenv("LLM_API_KEY"):
        raise ExperimentError(
            "真实 A/B 实验缺少 LLM_API_KEY"
        )

    prepared = build_plan(args)

    print("=" * 60)
    print("B3.2b A/B 预检")
    print("=" * 60)
    print(
        "目标表格:",
        prepared["plan"]["target"]["table_id"],
    )
    print(
        "Flat 行数:",
        prepared["plan"]["target"]["flat_row_count"],
    )
    print(
        "Logical 可信行数:",
        prepared["plan"]["target"][
            "logical_trusted_row_count"
        ],
    )
    print(
        "排除 needs_review:",
        prepared["plan"]["target"][
            "excluded_needs_review_ids"
        ],
    )
    print(
        "标准答案参数数:",
        prepared["gold"]["expected_parameter_count"],
    )
    print(
        "输出目录:",
        args.output_dir,
    )

    if not args.execute:
        if args.prepare_only:
            output_dir, plan_path, gold_path = (
                prepare_outputs(args, prepared)
            )
            print("已写入预检文件:")
            print(" -", plan_path)
            print(" -", gold_path)

        print(
            "安全预检模式：未调用真实 DeepSeek API。"
        )
        return 0

    output_dir, _, _ = prepare_outputs(
        args, prepared
    )

    flat_table = prepared["flat_table"]
    logical_table = prepared["logical_table"]

    flat_arm = run_arm(
        "flat",
        [copy.deepcopy(flat_table)],
        flat_table,
        prepared["logical_rows"],
        prepared["raw_structure"],
        prepared["pdf_sha256"],
        "flat",
        prepared["gold"],
    )
    logical_arm = run_arm(
        "logical",
        [copy.deepcopy(logical_table)],
        flat_table,
        prepared["logical_rows"],
        prepared["raw_structure"],
        prepared["pdf_sha256"],
        "logical",
        prepared["gold"],
    )

    comparison = build_comparison(
        flat_arm, logical_arm
    )
    recommendations = build_prompt_recommendations(
        flat_arm, logical_arm
    )

    result = {
        "schema": "ab_experiment_result/v1",
        "stage": "B3.2b",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_call": True,
        "plan": prepared["plan"],
        "gold_standard": prepared["gold"],
        "arms": [flat_arm, logical_arm],
        "comparison": comparison,
        "prompt_recommendations": recommendations,
        "checks": {
            "same_gold": True,
            "gold_record_count": prepared["gold"][
                "expected_parameter_count"
            ],
            "same_model": (
                flat_arm["extractor_config"]["model"]
                == logical_arm["extractor_config"][
                    "model"
                ]
            ),
            "same_temperature": (
                flat_arm["extractor_config"].get(
                    "temperature"
                )
                == logical_arm["extractor_config"].get(
                    "temperature"
                )
            ),
            "same_max_tokens": (
                flat_arm["extractor_config"][
                    "max_tokens"
                ]
                == logical_arm["extractor_config"][
                    "max_tokens"
                ]
            ),
            "same_chunking": (
                flat_arm["extractor_config"][
                    "rows_per_chunk"
                ]
                == logical_arm["extractor_config"][
                    "rows_per_chunk"
                ]
                and flat_arm["extractor_config"][
                    "max_split_depth"
                ]
                == logical_arm["extractor_config"][
                    "max_split_depth"
                ]
            ),
            "retry_called": False,
            "no_retry": True,
            "only_target_table": True,
        },
    }

    report = build_report_markdown(result)
    result["report_markdown"] = report

    result_path = write_json_exclusive(
        output_dir / "ab_result.json",
        result,
    )
    report_path = write_text_exclusive(
        output_dir / "ab_report.md",
        report,
    )

    print("A/B 实验完成:", result_path)
    print("A/B 报告:", report_path)

    for arm in result["arms"]:
        evaluation = arm["evaluation"]
        print(
            arm["arm"],
            {
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
                "matched": evaluation["matched_keys"],
                "strict": evaluation["strict_matched"],
                "tokens": arm["metrics"][
                    "total_tokens"
                ],
                "requests": arm["metrics"][
                    "request_count"
                ],
                "elapsed": arm["metrics"][
                    "elapsed_seconds"
                ],
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
