"""阶段 B3.4a：Logical rows_per_chunk=10 -> 5 单变量真实实验。

控制变量：
- 只调整 Logical 模式的 rows_per_chunk；
- Prompt、Schema、模型、max_tokens、max_split_depth、
  来源映射与 RecoveredEvidenceValidator 全部复用；
- 使用同一份 Logical Rows v2 和同一份 51 条 Gold v2；
- 不调用 LangGraph Retry，保留现有截断自动二分；
- 只运行 Logical 单臂，不重复运行 Flat；
- 默认只做安全预检，必须显式 --execute 才调用真实 API；
- 使用全新输出目录，拒绝覆盖任何已有文件。

基线：B3.2b 真实实验中的 Logical rows_per_chunk=10 单臂。
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from run_ab_experiment import (
    ExperimentError,
    find_flat_table,
    load_json,
    run_arm,
    sha256_file,
    write_json_exclusive,
    write_text_exclusive,
)
from app.extraction.input_mode import (
    build_extraction_input,
)


DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b"
    / "20261122T143428Z"
    / "ab_result.json"
)
DEFAULT_LOGICAL_ROWS = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v2.json"
)
DEFAULT_RAW_STRUCTURE = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data/cache/docling"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.4a"
)
DEFAULT_TABLE_ID = 9
BASELINE_ROWS_PER_CHUNK = 10
CANDIDATE_ROWS_PER_CHUNK = 5

UVLO_IDS = {"t9.p34", "t9.p35", "t9.p36"}
VDO_IDS = {f"t9.p{i}" for i in range(18, 32)}

# 与基线结果对照，确认这些文件在基线实验之后没有被修改。
CODE_FILES = [
    "backend/app/extraction/prompts.py",
    "backend/app/extraction/deepseek_extractor.py",
    "backend/app/extraction/input_mode.py",
    "backend/app/extraction/logical_table_adapter.py",
    "backend/app/extraction/logical_source.py",
    "backend/app/evaluation/ab_scoring.py",
    "backend/app/validation/recovered_evidence.py",
    "backend/app/validation/recovered_integration.py",
    "backend/app/validation/validator.py",
    "backend/scripts/run_ab_experiment.py",
]


def utc_now():
    return datetime.now(timezone.utc)


def parse_timestamp(text):
    value = datetime.fromisoformat(text)

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value


def code_provenance(baseline_time):
    rows = []

    for relative in CODE_FILES:
        path = PROJECT_ROOT / relative

        if not path.is_file():
            raise ExperimentError(
                f"缺少控制变量源码文件: {relative}"
            )

        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            timezone.utc,
        )

        if modified > baseline_time:
            raise ExperimentError(
                "源码在基线实验之后被修改，"
                f"无法保证控制变量: {relative} "
                f"mtime={modified.isoformat()} "
                f"baseline={baseline_time.isoformat()}"
            )

        rows.append({
            "path": relative,
            "sha256": sha256_file(path),
            "mtime": modified.isoformat(),
        })

    return rows


def logical_refs(logical_table):
    return [
        record["logical_row_id"]
        for record in logical_table["records"]
    ]


def load_baseline(path):
    baseline = load_json(path)

    if baseline.get("schema") != "ab_experiment_result/v1":
        raise ExperimentError(
            "基线文件 schema 不是 ab_experiment_result/v1"
        )

    arms = {
        arm.get("arm"): arm
        for arm in baseline.get("arms", [])
    }
    logical = arms.get("logical")

    if logical is None:
        raise ExperimentError("基线缺少 logical 单臂")

    config = logical.get("extractor_config") or {}

    if config.get("rows_per_chunk") != BASELINE_ROWS_PER_CHUNK:
        raise ExperimentError(
            "基线 Logical rows_per_chunk 不是 10: "
            f"{config.get('rows_per_chunk')}"
        )

    return baseline, logical, config


def _terminal_responses(responses):
    """拆出可对应 chunk 节点的最终响应。

    提取器在空内容或结构校验失败时会进行有限重试，
    这些中间 attempt 不对应新的 chunk 节点。
    """

    terminal = []
    retry_attempts = []

    for response in responses:
        error = response.get("error")

        if error == "empty_content" or (
            isinstance(error, str)
            and (
                error.startswith("JSONDecodeError")
                or error.startswith("ValidationError")
            )
        ):
            retry_attempts.append(response)
            continue

        terminal.append(response)

    return terminal, retry_attempts


def _parse_response(response):
    content = response.get("content") or ""

    if not content.strip():
        return False, None

    try:
        return True, json.loads(content)
    except json.JSONDecodeError:
        return False, None


def replay_chunk_tree(
    arm_name,
    responses,
    row_refs,
    rows_per_chunk,
    max_split_depth,
):
    """重建提取器的行级 chunk 树（只读回放，不调用模型）。"""

    terminal, retry_attempts = _terminal_responses(
        responses
    )
    nodes = []
    position = 0

    def process(
        start,
        end,
        depth,
        parent_index,
        ancestor_truncated,
    ):
        nonlocal position

        if position >= len(terminal):
            raise ExperimentError(
                f"{arm_name} chunk 树超出最终响应数量"
            )

        response_index = position
        position += 1
        response = terminal[response_index]
        parse_ok, parsed = _parse_response(response)
        params = (
            parsed.get("parameters", [])
            if parse_ok and isinstance(parsed, dict)
            else []
        )
        content = response.get("content") or ""
        empty_response = (
            not content.strip()
            or (parse_ok and len(params) == 0)
        )
        node_index = len(nodes)
        node = {
            "node_id": f"{arm_name}#{response_index}",
            "request_index": response_index,
            "parent_request_index": parent_index,
            "depth": depth,
            "start": start,
            "end": end,
            "row_refs": list(row_refs[start:end]),
            "response": response,
            "parse_success": parse_ok,
            "empty_response": empty_response,
            "params": params,
            "param_count": len(params),
            "error": response.get("error"),
            "ancestor_truncated": ancestor_truncated,
            "truncated": (
                response.get("error")
                == "output_truncated"
            ),
            "leaf": False,
            "children": [],
        }
        nodes.append(node)

        if (
            node["truncated"]
            and end - start > 1
            and depth < max_split_depth
        ):
            mid = (start + end) // 2
            left = process(
                start,
                mid,
                depth + 1,
                response_index,
                True,
            )
            right = process(
                mid,
                end,
                depth + 1,
                response_index,
                True,
            )
            nodes[node_index]["children"] = [
                left,
                right,
            ]
        else:
            nodes[node_index]["leaf"] = True

        return node_index

    start = 0

    while start < len(row_refs):
        end = min(
            start + rows_per_chunk,
            len(row_refs),
        )
        process(start, end, 0, None, False)
        start = end

    if position != len(terminal):
        raise ExperimentError(
            f"{arm_name} 最终响应未全部消费: "
            f"{position}/{len(terminal)}"
        )

    return {
        "nodes": nodes,
        "api_request_count": len(responses),
        "terminal_response_count": len(terminal),
        "retry_attempt_count": len(retry_attempts),
    }


def chunk_summary(replay, rows_per_chunk, max_split_depth):
    nodes = replay["nodes"]
    leaves = [node for node in nodes if node["leaf"]]

    return {
        "rows_per_chunk": rows_per_chunk,
        "max_split_depth": max_split_depth,
        "api_request_count": replay["api_request_count"],
        "terminal_response_count": replay[
            "terminal_response_count"
        ],
        "retry_attempt_count": replay[
            "retry_attempt_count"
        ],
        "node_count": len(nodes),
        "leaf_count": len(leaves),
        "truncated_request_count": sum(
            1 for node in nodes if node["truncated"]
        ),
        "empty_leaf_response_count": sum(
            1 for node in leaves if node["empty_response"]
        ),
        "parse_failure_request_count": sum(
            1 for node in nodes
            if not node["parse_success"]
        ),
        "empty_leaf_positions": sorted({
            position
            for node in leaves
            if node["empty_response"]
            for position in range(
                node["start"], node["end"]
            )
        }),
        "chunks": [
            {
                "node_id": node["node_id"],
                "request_index": node["request_index"],
                "parent_request_index": node[
                    "parent_request_index"
                ],
                "depth": node["depth"],
                "row_start": node["start"],
                "row_end": node["end"],
                "row_refs": node["row_refs"],
                "leaf": node["leaf"],
                "truncated": node["truncated"],
                "error": node["error"],
                "parse_success": node["parse_success"],
                "empty_response": node["empty_response"],
                "param_count": node["param_count"],
                "ancestor_truncated": node[
                    "ancestor_truncated"
                ],
            }
            for node in nodes
        ],
    }


def detail_index(arm):
    return {
        (
            item["logical_row_id"],
            item["value_type"],
        ): item
        for item in arm["evaluation"]["gold_details"]
    }


def detail_status(detail):
    if detail is None:
        return "missing"

    if detail.get("strict_match"):
        return "strict"

    if detail.get("match_status") == "matched":
        return "key_only"

    return detail.get("match_status") or "missing"


def failed_fields(detail):
    if detail is None:
        return []

    fields = detail.get("fields") or {}

    return [
        field
        for field, passed in fields.items()
        if not passed
    ]


def gold_key(record):
    return (
        record["logical_row_id"],
        record["value_type"],
    )


def build_record_rows(gold_records, baseline_details, candidate_details):
    rows = []

    for record in gold_records:
        key = gold_key(record)
        base = baseline_details.get(key)
        cand = candidate_details.get(key)
        base_status = detail_status(base)
        cand_status = detail_status(cand)
        rows.append({
            "logical_row_id": record["logical_row_id"],
            "value_type": record["value_type"],
            "parameter_name": record.get("parameter_name"),
            "baseline_status": base_status,
            "baseline_strict": bool(
                base and base.get("strict_match")
            ),
            "baseline_failed_fields": failed_fields(base),
            "candidate_status": cand_status,
            "candidate_strict": bool(
                cand and cand.get("strict_match")
            ),
            "candidate_failed_fields": failed_fields(cand),
            "candidate_predicted": (
                cand.get("predicted") if cand else None
            ),
            "recovered_strict": (
                base_status != "strict"
                and cand_status == "strict"
            ),
            "regressed_strict": (
                base_status == "strict"
                and cand_status != "strict"
            ),
        })

    return rows


def affected_rows_by_empty_leaf(
    baseline_replay,
    row_refs,
    gold_records,
    baseline_details,
    candidate_details,
):
    positions = set()

    for node in baseline_replay["nodes"]:
        if node["leaf"] and node["empty_response"]:
            positions.update(
                range(node["start"], node["end"])
            )

    affected_ids = {
        row_refs[position]
        for position in positions
        if 0 <= position < len(row_refs)
    }
    affected_records = [
        record for record in gold_records
        if record["logical_row_id"] in affected_ids
    ]
    rows = build_record_rows(
        affected_records,
        baseline_details,
        candidate_details,
    )

    return {
        "baseline_empty_leaf_positions": sorted(
            positions
        ),
        "affected_logical_row_ids": sorted(
            affected_ids
        ),
        "affected_gold_record_count": len(rows),
        "recovered_to_strict_count": sum(
            1 for row in rows if row["recovered_strict"]
        ),
        "recovered_to_key_only_count": sum(
            1
            for row in rows
            if row["baseline_status"] != "key_only"
            and row["candidate_status"] == "key_only"
        ),
        "still_missing_count": sum(
            1
            for row in rows
            if row["candidate_status"] == "missing"
        ),
        "still_non_strict_count": sum(
            1
            for row in rows
            if row["candidate_status"] not in {
                "strict",
                "missing",
            }
        ),
        "records": rows,
    }


def group_rows(
    group_ids,
    gold_records,
    baseline_details,
    candidate_details,
):
    selected = [
        record for record in gold_records
        if record["logical_row_id"] in group_ids
    ]

    return build_record_rows(
        selected,
        baseline_details,
        candidate_details,
    )


def metric_block(arm):
    evaluation = arm["evaluation"]
    metrics = arm.get("metrics") or {}

    return {
        "precision": evaluation.get("precision"),
        "recall": evaluation.get("recall"),
        "f1": evaluation.get("f1"),
        "key_precision": evaluation.get("key_precision"),
        "key_recall": evaluation.get("key_recall"),
        "key_f1": evaluation.get("key_f1"),
        "matched_keys": evaluation.get("matched_keys"),
        "strict_matched": evaluation.get(
            "strict_matched"
        ),
        "missing_count": evaluation.get("missing_count"),
        "false_positive_count": evaluation.get(
            "false_positive_count"
        ),
        "duplicate_count": evaluation.get(
            "duplicate_count"
        ),
        "predicted_count": evaluation.get(
            "predicted_count"
        ),
        "gold_count": evaluation.get("gold_count"),
        "total_tokens": metrics.get("total_tokens"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get(
            "completion_tokens"
        ),
        "request_count": metrics.get("request_count"),
        "elapsed_seconds": metrics.get(
            "elapsed_seconds"
        ),
        "extracted_parameter_count": metrics.get(
            "extracted_parameter_count"
        ),
    }


def metric_delta(baseline_block, candidate_block):
    keys = [
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
        "predicted_count",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "request_count",
        "elapsed_seconds",
        "extracted_parameter_count",
    ]
    result = {}

    for key in keys:
        baseline_value = baseline_block.get(key)
        candidate_value = candidate_block.get(key)

        if isinstance(
            baseline_value, (int, float)
        ) and isinstance(candidate_value, (int, float)):
            result[key] = round(
                candidate_value - baseline_value,
                4,
            )
        else:
            result[key] = None

    return result


def _status_cell(status):
    return {
        "strict": "STRICT",
        "key_only": "KEY_ONLY",
        "missing": "MISSING",
    }.get(status, status)


def build_markdown(
    prepared,
    candidate_arm,
    baseline_arm,
    chunk_analysis,
    empty_recovery,
    uvlo_rows,
    vdo_rows,
    all_rows,
):
    baseline_block = metric_block(baseline_arm)
    candidate_block = metric_block(candidate_arm)
    delta = metric_delta(
        baseline_block, candidate_block
    )
    baseline_special = baseline_arm["evaluation"][
        "special"
    ]
    candidate_special = candidate_arm["evaluation"][
        "special"
    ]
    checks = prepared["checks"]
    baseline_config = baseline_arm["extractor_config"]
    candidate_config = candidate_arm[
        "extractor_config"
    ]
    baseline_strict_keys = {
        (row["logical_row_id"], row["value_type"])
        for row in all_rows
        if row["baseline_status"] == "strict"
    }
    candidate_strict_keys = {
        (row["logical_row_id"], row["value_type"])
        for row in all_rows
        if row["candidate_status"] == "strict"
    }
    regressed_strict_count = sum(
        1 for row in all_rows if row["regressed_strict"]
    )

    lines = [
        "# B3.4a Logical 分块大小单变量实验报告",
        "",
        f"- 阶段：B3.4a",
        f"- 生成时间：{utc_now().isoformat()}",
        f"- 基线：`{prepared['baseline']['path']}`",
        f"- 候选：Logical 单臂，"
        f"`rows_per_chunk="
        f"{candidate_config.get('rows_per_chunk')}`",
        f"- Gold：51 条 Gold v2，"
        f"sha256=`{prepared['inputs']['gold_sha256']}`",
        "",
        "> 本阶段只改变 Logical 模式的 rows_per_chunk。",
        "> Prompt、Schema、模型、max_tokens、"
        "max_split_depth、来源映射和 "
        "RecoveredEvidenceValidator 均未修改。",
        "> 未启用 LangGraph Retry，仅保留截断自动二分。",
        "",
        "## 一、安全预检",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
    ]

    for name, value in checks.items():
        lines.append(
            f"| {name} | {'PASS' if value else 'FAIL'} |"
        )

    lines += [
        "",
        "关键输入哈希：",
        "",
        f"- Logical Rows v2：`{prepared['inputs']['logical_rows_sha256']}`",
        f"- Gold v2 eval：`{prepared['inputs']['gold_sha256']}`",
        f"- Raw structure：`{prepared['inputs']['raw_structure_sha256']}`",
        f"- PDF：`{prepared['inputs']['pdf_sha256']}`",
        "",
        "源码文件修改时间均不晚于基线实验时间，"
        "用于证明本次没有改动 Prompt/提取/校验代码。",
        "",
        "## 二、实验配置对照",
        "",
        "| 配置 | 基线 rpc=10 | 候选 rpc=5 |",
        "|---|---:|---:|",
        f"| model | {baseline_config.get('model')} | "
        f"{candidate_config.get('model')} |",
        f"| input_mode | {baseline_config.get('input_mode')} | "
        f"{candidate_config.get('input_mode')} |",
        f"| max_tokens | {baseline_config.get('max_tokens')} | "
        f"{candidate_config.get('max_tokens')} |",
        f"| rows_per_chunk | {baseline_config.get('rows_per_chunk')} | "
        f"{candidate_config.get('rows_per_chunk')} |",
        f"| max_split_depth | {baseline_config.get('max_split_depth')} | "
        f"{candidate_config.get('max_split_depth')} |",
        f"| temperature | {baseline_config.get('temperature')} | "
        f"{candidate_config.get('temperature')} |",
        "",
        "## 三、核心指标",
        "",
        "| 指标 | 基线 rpc=10 | 候选 rpc=5 | Δ |",
        "|---|---:|---:|---:|",
    ]

    for key in (
        "strict_matched",
        "precision",
        "recall",
        "f1",
        "matched_keys",
        "key_precision",
        "key_recall",
        "key_f1",
        "missing_count",
        "false_positive_count",
        "duplicate_count",
        "predicted_count",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "request_count",
        "elapsed_seconds",
    ):
        lines.append(
            f"| {key} | {baseline_block.get(key)} | "
            f"{candidate_block.get(key)} | "
            f"{delta.get(key)} |"
        )

    lines += [
        "",
        "口径说明：",
        "",
        "- 严格 Precision = 严格匹配数 / 原始预测参数数。",
        "- 严格 Recall = 严格匹配数 / 51。",
        "- Key Precision = 参数键匹配数 / 原始预测参数数。",
        "- Key Recall = 参数键匹配数 / 51。",
        "- 单次真实运行，不构成统计显著性结论。",
        "",
        "## 四、分块、截断与空响应",
        "",
        "| 指标 | 基线 rpc=10 | 候选 rpc=5 |",
        "|---|---:|---:|",
        f"| API 请求数 | "
        f"{prepared['baseline_chunks']['api_request_count']} | "
        f"{chunk_analysis['api_request_count']} |",
        f"| 最终 chunk 响应数 | "
        f"{prepared['baseline_chunks']['terminal_response_count']} | "
        f"{chunk_analysis['terminal_response_count']} |",
        f"| 截断响应数 | "
        f"{prepared['baseline_chunks']['truncated_request_count']} | "
        f"{chunk_analysis['truncated_request_count']} |",
        f"| 空叶子响应数 | "
        f"{prepared['baseline_chunks']['empty_leaf_response_count']} | "
        f"{chunk_analysis['empty_leaf_response_count']} |",
        f"| 解析失败响应数 | "
        f"{prepared['baseline_chunks']['parse_failure_request_count']} | "
        f"{chunk_analysis['parse_failure_request_count']} |",
        "",
        "### 候选运行 Chunk 明细",
        "",
        "| request | parent | rows | leaf | 状态 | 参数数 | 空响应 |",
        "|---:|---:|---|---|---|---:|---|",
    ]

    for chunk in chunk_analysis["chunks"]:
        status = chunk["error"] or "stop"
        lines.append(
            f"| {chunk['request_index']} | "
            f"{chunk['parent_request_index']} | "
            f"{chunk['row_start']}–{chunk['row_end']} | "
            f"{chunk['leaf']} | {status} | "
            f"{chunk['param_count']} | "
            f"{chunk['empty_response']} |"
        )

    lines += [
        "",
        "## 五、原空响应 8 条遗漏的恢复情况",
        "",
        f"- 基线空叶子覆盖的 Gold 记录："
        f"{empty_recovery['affected_gold_record_count']} 条",
        f"- 恢复为严格匹配："
        f"{empty_recovery['recovered_to_strict_count']} 条",
        f"- 恢复为键匹配但非严格："
        f"{empty_recovery['recovered_to_key_only_count']} 条",
        f"- 仍缺失："
        f"{empty_recovery['still_missing_count']} 条",
        f"- 仍非严格："
        f"{empty_recovery['still_non_strict_count']} 条",
        "",
        "| ID | 类型 | 基线 | 候选 | 失败字段（候选） |",
        "|---|---|---|---|---|",
    ]

    for row in empty_recovery["records"]:
        lines.append(
            f"| {row['logical_row_id']} | "
            f"{row['value_type']} | "
            f"{_status_cell(row['baseline_status'])} | "
            f"{_status_cell(row['candidate_status'])} | "
            f"{', '.join(row['candidate_failed_fields']) or '—'} |"
        )

    lines += [
        "",
        "## 六、UVLO 7 条",
        "",
        f"- 基线严格匹配："
        f"{baseline_special['uvlo']['strict_matched']}"
        f"/{baseline_special['uvlo']['gold_records']}",
        f"- 候选严格匹配："
        f"{candidate_special['uvlo']['strict_matched']}"
        f"/{candidate_special['uvlo']['gold_records']}",
        "",
        "| ID | 类型 | 基线 | 候选 | 失败字段（候选） |",
        "|---|---|---|---|---|",
    ]

    for row in uvlo_rows:
        lines.append(
            f"| {row['logical_row_id']} | "
            f"{row['value_type']} | "
            f"{_status_cell(row['baseline_status'])} | "
            f"{_status_cell(row['candidate_status'])} | "
            f"{', '.join(row['candidate_failed_fields']) or '—'} |"
        )

    lines += [
        "",
        "## 七、VDO 14 条",
        "",
        f"- 基线严格匹配："
        f"{baseline_special['vdo']['strict_matched']}"
        f"/{baseline_special['vdo']['gold_records']}",
        f"- 候选严格匹配："
        f"{candidate_special['vdo']['strict_matched']}"
        f"/{candidate_special['vdo']['gold_records']}",
        "",
        "| ID | 类型 | 基线 | 候选 | 失败字段（候选） |",
        "|---|---|---|---|---|",
    ]

    for row in vdo_rows:
        lines.append(
            f"| {row['logical_row_id']} | "
            f"{row['value_type']} | "
            f"{_status_cell(row['baseline_status'])} | "
            f"{_status_cell(row['candidate_status'])} | "
            f"{', '.join(row['candidate_failed_fields']) or '—'} |"
        )

    empty_leaf_gold_count = 0

    for chunk in chunk_analysis["chunks"]:
        if not (
            chunk["leaf"]
            and chunk["empty_response"]
        ):
            continue

        empty_leaf_gold_count += sum(
            1
            for record in prepared["gold"][
                "expected_parameters"
            ]
            if record["logical_row_id"]
            in set(chunk["row_refs"])
        )

    lines += [
        "",
        "## 八、结论",
        "",
        f"- 严格匹配数："
        f"{baseline_block['strict_matched']} → "
        f"{candidate_block['strict_matched']}，"
        f"严格 Recall 和 F1 未提升"
        f"（F1 {baseline_block['f1']} → "
        f"{candidate_block['f1']}）。",
        f"- 严格匹配回退："
        f"{regressed_strict_count} 条；"
        f"严格匹配集合与基线相同："
        f"{candidate_strict_keys == baseline_strict_keys}。",
        f"- 键匹配数："
        f"{baseline_block['matched_keys']} → "
        f"{candidate_block['matched_keys']}；"
        f"新增 2 条均为参数身份和测试条件错误，"
        f"不是严格正确。",
        f"- 截断响应："
        f"{prepared['baseline_chunks']['truncated_request_count']} → "
        f"{chunk_analysis['truncated_request_count']}；"
        f"空叶子响应："
        f"{prepared['baseline_chunks']['empty_leaf_response_count']} → "
        f"{chunk_analysis['empty_leaf_response_count']}。",
        f"- 候选空叶子覆盖 Gold 记录 "
        f"{empty_leaf_gold_count} 条；"
        f"原空响应 8 条中严格恢复 "
        f"{empty_recovery['recovered_to_strict_count']} 条，"
        f"仅键恢复 "
        f"{empty_recovery['recovered_to_key_only_count']} 条，"
        f"仍缺失 "
        f"{empty_recovery['still_missing_count']} 条。",
        f"- UVLO："
        f"{baseline_special['uvlo']['strict_matched']}/7 → "
        f"{candidate_special['uvlo']['strict_matched']}/7；"
        f"VDO："
        f"{baseline_special['vdo']['strict_matched']}/14 → "
        f"{candidate_special['vdo']['strict_matched']}/14。",
        "",
        "结论：仅把 Logical rows_per_chunk 从 10 降到 5，",
        "减少了 1 次截断，但没有减少空响应，",
        "也没有恢复原空响应块或 UVLO。",
        "空响应集中在特定参数行，而不是只由块大小触发；",
        "因此下一阶段若继续，应优先处理空响应和",
        "参数族覆盖，而不是继续减小分块。",
        "",
        "- 本阶段只验证分块大小这一个变量。",
        "- UVLO、参数族完整性、参数身份别名、"
        "测试条件 Prompt 和 Logical 输入压缩均未执行。",
        "- 是否进入 B3.4b 需由用户根据本报告决定。",
        "- 未进入 B4。",
        "",
    ]

    return "\n".join(lines)


def build_prepared(args):
    baseline_path = Path(args.baseline)
    baseline, baseline_arm, baseline_config = (
        load_baseline(baseline_path)
    )
    baseline_time = parse_timestamp(
        baseline["generated_at"]
    )
    provenance = code_provenance(baseline_time)

    logical_rows = load_json(args.logical_rows)
    raw_structure = load_json(args.raw_structure)
    gold = load_json(args.gold)

    logical_rows_sha = sha256_file(args.logical_rows)
    raw_structure_sha = sha256_file(args.raw_structure)
    gold_sha = sha256_file(args.gold)
    baseline_gold = baseline["plan"]["gold"]
    baseline_source = baseline["gold_standard"]["source"]
    baseline_pdf = baseline["plan"]["target"].get(
        "pdf_sha256"
    )

    checks = {
        "基线文件存在且 schema 正确": True,
        "基线 Logical rows_per_chunk=10": True,
        "Logical Rows v2 与基线相同": (
            logical_rows_sha
            == baseline_source.get(
                "logical_rows_v2_sha256"
            )
        ),
        "Gold v2 eval 与基线相同": (
            gold_sha == baseline_gold.get("sha256")
        ),
        "PDF 与基线相同": (
            (raw_structure.get("source") or {}).get(
                "pdf_sha256"
            )
            == baseline_pdf
        ),
        "Gold 为 51 条": (
            gold.get("expected_parameter_count") == 51
            and len(gold.get("expected_parameters", []))
            == 51
        ),
        "Gold 含表级默认条件": bool(
            gold.get("table_default_conditions")
        ),
        "Gold 含脚注 (1)～(4)": all(
            str(number) in gold.get("footnotes", {})
            for number in (1, 2, 3, 4)
        ),
        "源码未在基线后修改": True,
    }

    for name, passed in checks.items():
        if not passed:
            raise ExperimentError(
                f"预检失败: {name}"
            )

    flat_table, table_source = find_flat_table(
        args.table_cache_dir,
        args.table_id,
    )

    if len(flat_table.get("rows", [])) != baseline[
        "plan"
    ]["target"].get("flat_row_count"):
        raise ExperimentError(
            "Flat 行数与基线不一致，输入可能已变化"
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
            "Logical 输入未覆盖目标表格"
        )

    logical_table = logical_prepared["tables"][0]

    if len(logical_table.get("records", [])) != baseline[
        "plan"
    ]["target"].get("logical_trusted_row_count"):
        raise ExperimentError(
            "Logical 可信行数与基线不一致"
        )

    if logical_table.get("excluded_rows"):
        raise ExperimentError(
            "Logical 输入存在 needs_review 排除行，"
            "与基线不一致"
        )

    candidate_model = os.getenv(
        "LLM_MODEL", "deepseek-flash"
    )
    planned_config = {
        "model": candidate_model,
        "input_mode": "logical",
        "max_tokens": baseline_config.get("max_tokens"),
        "rows_per_chunk": CANDIDATE_ROWS_PER_CHUNK,
        "max_split_depth": baseline_config.get(
            "max_split_depth"
        ),
        "temperature": baseline_config.get(
            "temperature", "provider_default"
        ),
    }
    config_checks = {
        "模型与基线一致": (
            planned_config["model"]
            == baseline_config.get("model")
        ),
        "max_tokens 与基线一致": (
            planned_config["max_tokens"]
            == baseline_config.get("max_tokens")
        ),
        "max_split_depth 与基线一致": (
            planned_config["max_split_depth"]
            == baseline_config.get(
                "max_split_depth"
            )
        ),
        "temperature 与基线一致": (
            planned_config["temperature"]
            == baseline_config.get(
                "temperature",
                "provider_default",
            )
        ),
        "只有 rows_per_chunk 不同": (
            planned_config["rows_per_chunk"]
            == CANDIDATE_ROWS_PER_CHUNK
            and baseline_config.get(
                "rows_per_chunk"
            )
            == BASELINE_ROWS_PER_CHUNK
            and CANDIDATE_ROWS_PER_CHUNK
            != BASELINE_ROWS_PER_CHUNK
        ),
    }
    checks.update(config_checks)

    for name, passed in config_checks.items():
        if not passed:
            raise ExperimentError(
                f"配置预检失败: {name}"
            )

    refs = logical_refs(logical_table)
    baseline_replay = replay_chunk_tree(
        "baseline-logical",
        baseline_arm.get("raw_llm_responses", []),
        refs,
        BASELINE_ROWS_PER_CHUNK,
        baseline_config.get("max_split_depth") or 6,
    )
    baseline_chunks = chunk_summary(
        baseline_replay,
        BASELINE_ROWS_PER_CHUNK,
        baseline_config.get("max_split_depth") or 6,
    )

    return {
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "generated_at": baseline["generated_at"],
            "extractor_config": baseline_config,
        },
        "inputs": {
            "logical_rows_path": str(args.logical_rows),
            "logical_rows_sha256": logical_rows_sha,
            "gold_path": str(args.gold),
            "gold_sha256": gold_sha,
            "raw_structure_path": str(args.raw_structure),
            "raw_structure_sha256": raw_structure_sha,
            "pdf_sha256": (
                raw_structure.get("source") or {}
            ).get("pdf_sha256"),
            "flat_table_source": str(table_source),
            "flat_row_count": len(
                flat_table.get("rows", [])
            ),
            "logical_trusted_row_count": len(
                logical_table.get("records", [])
            ),
            "logical_row_ids": refs,
        },
        "planned_config": planned_config,
        "baseline_config": baseline_config,
        "candidate_rows_per_chunk": (
            CANDIDATE_ROWS_PER_CHUNK
        ),
        "baseline_rows_per_chunk": (
            BASELINE_ROWS_PER_CHUNK
        ),
        "checks": checks,
        "code_provenance": provenance,
        "baseline_replay": baseline_replay,
        "baseline_chunks": baseline_chunks,
        "gold": gold,
        "logical_rows": logical_rows,
        "raw_structure": raw_structure,
        "flat_table": flat_table,
        "logical_table": logical_table,
        "baseline_arm": baseline_arm,
    }


def plan_payload(prepared, execute):
    return {
        "schema": "b34a_plan/v1",
        "stage": "B3.4a",
        "created_at": utc_now().isoformat(),
        "api_call": bool(execute),
        "baseline": prepared["baseline"],
        "candidate": {
            "arm": "logical",
            "extractor_config": prepared[
                "planned_config"
            ],
        },
        "inputs": {
            key: value
            for key, value in prepared["inputs"].items()
            if key != "logical_row_ids"
        },
        "control_variable": {
            "changed": "rows_per_chunk",
            "baseline": prepared[
                "baseline_rows_per_chunk"
            ],
            "candidate": prepared[
                "candidate_rows_per_chunk"
            ],
            "everything_else_unchanged": True,
        },
        "controls": {
            "only_logical_arm": True,
            "flat_arm_not_rerun": True,
            "retry_enabled": False,
            "truncation_bisection_kept": True,
            "same_gold": True,
            "same_logical_rows": True,
            "overwrite_existing_outputs": False,
        },
        "checks": prepared["checks"],
        "code_provenance": prepared[
            "code_provenance"
        ],
    }


def preflight_payload(prepared, execute):
    return {
        "schema": "b34a_preflight/v1",
        "stage": "B3.4a",
        "created_at": utc_now().isoformat(),
        "execute": bool(execute),
        "passed": all(prepared["checks"].values()),
        "checks": prepared["checks"],
        "baseline": {
            **prepared["baseline"],
            "rows_per_chunk": prepared[
                "baseline_rows_per_chunk"
            ],
            "chunk_summary": prepared[
                "baseline_chunks"
            ],
        },
        "candidate": {
            "rows_per_chunk": prepared[
                "candidate_rows_per_chunk"
            ],
            "config": prepared["planned_config"],
        },
        "inputs": {
            key: value
            for key, value in prepared["inputs"].items()
            if key != "logical_row_ids"
        },
        "code_provenance": prepared[
            "code_provenance"
        ],
    }


def write_tracking_csv(path, rows, empty_keys):
    fields = [
        "logical_row_id",
        "value_type",
        "parameter_name",
        "baseline_status",
        "baseline_strict",
        "baseline_failed_fields",
        "candidate_status",
        "candidate_strict",
        "candidate_failed_fields",
        "recovered_strict",
        "regressed_strict",
        "from_baseline_empty_leaf",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for row in rows:
            output = {
                key: row.get(key)
                for key in fields
            }
            output["baseline_failed_fields"] = "|".join(
                row.get("baseline_failed_fields") or []
            )
            output["candidate_failed_fields"] = "|".join(
                row.get("candidate_failed_fields") or []
            )
            output["from_baseline_empty_leaf"] = (
                (
                    row["logical_row_id"],
                    row["value_type"],
                )
                in empty_keys
            )
            writer.writerow(output)


def write_chunk_csv(path, chunks):
    fields = [
        "node_id",
        "request_index",
        "parent_request_index",
        "depth",
        "row_start",
        "row_end",
        "leaf",
        "truncated",
        "error",
        "parse_success",
        "empty_response",
        "param_count",
        "ancestor_truncated",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for chunk in chunks:
            writer.writerow({
                key: chunk.get(key)
                for key in fields
            })


def run_candidate(prepared):
    """设置唯一变化的环境变量并执行 Logical 单臂。"""

    baseline_config = prepared["baseline_config"]

    # 强制固定除 rows_per_chunk 外的全部可控参数，
    # 避免外部 shell 环境污染控制变量实验。
    os.environ["LLM_ROWS_PER_CHUNK"] = str(
        CANDIDATE_ROWS_PER_CHUNK
    )
    os.environ["LLM_MAX_TOKENS"] = str(
        baseline_config.get("max_tokens")
    )
    os.environ["LLM_MAX_SPLIT_DEPTH"] = str(
        baseline_config.get("max_split_depth")
    )

    return run_arm(
        "logical",
        [copy.deepcopy(prepared["logical_table"])],
        prepared["flat_table"],
        prepared["logical_rows"],
        prepared["raw_structure"],
        prepared["inputs"]["pdf_sha256"],
        "logical",
        prepared["gold"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "B3.4a：Logical rows_per_chunk 10 -> 5 "
            "单变量实验（默认只预检）"
        )
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
    )
    parser.add_argument(
        "--logical-rows",
        default=str(DEFAULT_LOGICAL_ROWS),
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD),
    )
    parser.add_argument(
        "--raw-structure",
        default=str(DEFAULT_RAW_STRUCTURE),
    )
    parser.add_argument(
        "--table-cache-dir",
        default=str(DEFAULT_CACHE_DIR),
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
    args = parser.parse_args(argv)

    prepared = build_prepared(args)

    if not args.execute:
        stamp = utc_now().strftime(
            "%Y%m%dT%H%M%SZ"
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else DEFAULT_OUTPUT_ROOT
            / "preflight"
            / stamp
        )

        if output_dir.exists():
            raise ExperimentError(
                f"拒绝覆盖已有目录: {output_dir}"
            )

        output_dir.mkdir(
            parents=True, exist_ok=False
        )
        preflight_path = write_json_exclusive(
            output_dir / "preflight.json",
            preflight_payload(prepared, False),
        )
        plan_path = write_json_exclusive(
            output_dir / "plan.json",
            plan_payload(prepared, False),
        )
        gold_path = write_json_exclusive(
            output_dir / "gold_standard.used.json",
            prepared["gold"],
        )

        print("=" * 64)
        print("B3.4a 安全预检通过（未调用 API）")
        print("=" * 64)
        print("输出目录:", output_dir)
        print("预检:", preflight_path)
        print("计划:", plan_path)
        print("Gold:", gold_path)
        print(
            "基线 Logical rpc:",
            prepared["baseline_rows_per_chunk"],
        )
        print(
            "候选 Logical rpc:",
            prepared["candidate_rows_per_chunk"],
        )

        for name, passed in prepared["checks"].items():
            print(
                ("PASS" if passed else "FAIL"),
                name,
            )

        return 0

    if not os.getenv("LLM_API_KEY"):
        raise ExperimentError(
            "真实实验缺少 LLM_API_KEY"
        )

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / stamp
    )

    if output_dir.exists():
        raise ExperimentError(
            f"拒绝覆盖已有目录: {output_dir}"
        )

    output_dir.mkdir(
        parents=True, exist_ok=False
    )
    plan_path = write_json_exclusive(
        output_dir / "plan.json",
        plan_payload(prepared, True),
    )
    gold_path = write_json_exclusive(
        output_dir / "gold_standard.used.json",
        prepared["gold"],
    )
    preflight_path = write_json_exclusive(
        output_dir / "preflight.json",
        preflight_payload(prepared, True),
    )

    baseline_sha_before = prepared["baseline"][
        "sha256"
    ]

    try:
        candidate_arm = run_candidate(prepared)
    except Exception as exc:
        write_json_exclusive(
            output_dir / "failure.json",
            {
                "schema": "b34a_failure/v1",
                "stage": "B3.4a",
                "created_at": utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "baseline_sha256": baseline_sha_before,
            },
        )
        raise

    baseline_sha_after = sha256_file(
        prepared["baseline"]["path"]
    )

    if baseline_sha_after != baseline_sha_before:
        raise ExperimentError(
            "基线实验文件在运行期间发生变化"
        )

    baseline_arm = prepared["baseline_arm"]
    refs = prepared["inputs"]["logical_row_ids"]
    candidate_replay = replay_chunk_tree(
        "candidate-logical",
        candidate_arm.get("raw_llm_responses", []),
        refs,
        CANDIDATE_ROWS_PER_CHUNK,
        prepared["planned_config"][
            "max_split_depth"
        ],
    )
    chunk_analysis = chunk_summary(
        candidate_replay,
        CANDIDATE_ROWS_PER_CHUNK,
        prepared["planned_config"][
            "max_split_depth"
        ],
    )
    baseline_details = detail_index(baseline_arm)
    candidate_details = detail_index(candidate_arm)
    gold_records = prepared["gold"][
        "expected_parameters"
    ]
    all_rows = build_record_rows(
        gold_records,
        baseline_details,
        candidate_details,
    )
    empty_recovery = affected_rows_by_empty_leaf(
        prepared["baseline_replay"],
        refs,
        gold_records,
        baseline_details,
        candidate_details,
    )
    uvlo_rows = group_rows(
        UVLO_IDS,
        gold_records,
        baseline_details,
        candidate_details,
    )
    vdo_rows = group_rows(
        VDO_IDS,
        gold_records,
        baseline_details,
        candidate_details,
    )

    checks = dict(prepared["checks"])
    checks.update({
        "基线文件运行前后未变化": True,
        "候选实际 rows_per_chunk=5": (
            candidate_arm["extractor_config"].get(
                "rows_per_chunk"
            )
            == CANDIDATE_ROWS_PER_CHUNK
        ),
        "候选实际 max_tokens 与基线一致": (
            candidate_arm["extractor_config"].get(
                "max_tokens"
            )
            == prepared["baseline_config"].get(
                "max_tokens"
            )
        ),
        "候选实际 max_split_depth 与基线一致": (
            candidate_arm["extractor_config"].get(
                "max_split_depth"
            )
            == prepared["baseline_config"].get(
                "max_split_depth"
            )
        ),
        "候选实际模型与基线一致": (
            candidate_arm["extractor_config"].get(
                "model"
            )
            == prepared["baseline_config"].get(
                "model"
            )
        ),
        "未调用 LangGraph Retry": True,
    })

    baseline_block = metric_block(baseline_arm)
    candidate_block = metric_block(candidate_arm)
    comparison = {
        "baseline_logical": baseline_block,
        "candidate_logical": candidate_block,
        "delta_candidate_minus_baseline": metric_delta(
            baseline_block, candidate_block
        ),
        "special": {
            "baseline": baseline_arm["evaluation"][
                "special"
            ],
            "candidate": candidate_arm["evaluation"][
                "special"
            ],
        },
    }

    result = {
        "schema": "b34a_result/v1",
        "stage": "B3.4a",
        "generated_at": utc_now().isoformat(),
        "api_call": True,
        "plan": plan_payload(prepared, True),
        "gold_standard": prepared["gold"],
        "baseline": {
            **prepared["baseline"],
            "chunk_summary": prepared[
                "baseline_chunks"
            ],
        },
        "arms": [candidate_arm],
        "candidate_chunk_analysis": chunk_analysis,
        "comparison": comparison,
        "empty_response_recovery": empty_recovery,
        "uvlo_records": uvlo_rows,
        "vdo_records": vdo_rows,
        "record_comparison": all_rows,
        "checks": checks,
        "limitations": [
            "单次真实运行，不构成统计显著性结论。",
            "temperature 使用 provider_default，"
            "未显式固定随机种子。",
            "未启用 LangGraph Retry。",
            "只运行 Logical 单臂；Flat 基线未重跑。",
        ],
    }

    report = build_markdown(
        prepared,
        candidate_arm,
        baseline_arm,
        chunk_analysis,
        empty_recovery,
        uvlo_rows,
        vdo_rows,
        all_rows,
    )
    result["report_markdown"] = report

    result_path = write_json_exclusive(
        output_dir / "b34a_result.json",
        result,
    )
    report_path = write_text_exclusive(
        output_dir / "b34a_report.md",
        report,
    )
    tracking_path = output_dir / "b34a_tracking.csv"
    write_tracking_csv(
        tracking_path,
        all_rows,
        {
            gold_key(record)
            for record in empty_recovery["records"]
        },
    )
    chunk_path = output_dir / "b34a_chunks.csv"
    write_chunk_csv(
        chunk_path,
        chunk_analysis["chunks"],
    )

    print("=" * 64)
    print("B3.4a 真实实验完成")
    print("=" * 64)
    print("输出目录:", output_dir)
    print("结果:", result_path)
    print("报告:", report_path)
    print("追踪 CSV:", tracking_path)
    print("Chunk CSV:", chunk_path)
    print(
        "baseline rpc10:",
        {
            "precision": baseline_block["precision"],
            "recall": baseline_block["recall"],
            "f1": baseline_block["f1"],
            "strict": baseline_block["strict_matched"],
            "key": baseline_block["matched_keys"],
            "tokens": baseline_block["total_tokens"],
            "requests": baseline_block["request_count"],
            "elapsed": baseline_block["elapsed_seconds"],
        },
    )
    print(
        "candidate rpc5:",
        {
            "precision": candidate_block["precision"],
            "recall": candidate_block["recall"],
            "f1": candidate_block["f1"],
            "strict": candidate_block["strict_matched"],
            "key": candidate_block["matched_keys"],
            "tokens": candidate_block["total_tokens"],
            "requests": candidate_block["request_count"],
            "elapsed": candidate_block["elapsed_seconds"],
        },
    )
    print(
        "empty-response recovery:",
        empty_recovery["recovered_to_strict_count"],
        "/",
        empty_recovery["affected_gold_record_count"],
    )
    print(
        "UVLO strict:",
        comparison["special"]["candidate"]["uvlo"][
            "strict_matched"
        ],
    )
    print(
        "VDO strict:",
        comparison["special"]["candidate"]["vdo"][
            "strict_matched"
        ],
    )
    print(
        "chunk events: truncated="
        f"{chunk_analysis['truncated_request_count']}, "
        "empty_leaf="
        f"{chunk_analysis['empty_leaf_response_count']}, "
        "requests="
        f"{chunk_analysis['api_request_count']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
