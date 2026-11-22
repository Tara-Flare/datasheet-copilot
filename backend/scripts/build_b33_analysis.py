"""阶段 B3.3：A/B 实验遗漏归因与 51 条 Gold 追踪。

只读取已有 B3.2b 实验产物，不调用 DeepSeek API，
不修改任何现有实验文件。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.ab_scoring import (
    normalize_conditions,
    normalize_unit,
)
from app.extraction.logical_table_adapter import (
    adapt_logical_rows,
)


DEFAULT_RESULT = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b"
    / "20261122T143428Z"
    / "ab_result.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.3"
)

FAILURE_CATEGORIES = [
    "未生成",
    "输出截断",
    "空响应",
    "解析失败",
    "来源映射失败",
    "参数身份错误",
    "测试条件缺失",
    "单位或数值错误",
]


class AnalysisError(ValueError):
    """诊断分析失败。"""


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def normalize_number(value):
    text = str(value or "").strip()

    return (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )


def to_decimal(value):
    text = normalize_number(value)

    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def gold_key(item):
    return (
        item["logical_row_id"],
        item["value_type"],
    )


def parse_raw_content(response):
    content = response.get("content") or ""

    if not content.strip():
        return False, None

    try:
        return True, json.loads(content)
    except json.JSONDecodeError:
        return False, None


def build_chunk_tree(
    arm_name,
    responses,
    row_refs,
    rows_per_chunk,
    max_split_depth,
):
    """重建 DeepSeekExtractor 的行级二分 chunk 树。"""

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

        if position >= len(responses):
            raise AnalysisError(
                f"{arm_name} chunk 树超出 raw responses 数量"
            )

        response_index = position
        position += 1
        response = responses[response_index]
        parsed_ok, parsed = parse_raw_content(
            response
        )
        params = (
            parsed.get("parameters", [])
            if parsed_ok and isinstance(parsed, dict)
            else []
        )
        node_index = len(nodes)
        error = response.get("error")
        content = response.get("content") or ""
        empty_response = (
            not content.strip()
            or (
                parsed_ok
                and len(params) == 0
            )
        )
        node = {
            "node_id": f"{arm_name}#{response_index}",
            "arm": arm_name,
            "request_index": response_index,
            "parent_request_index": parent_index,
            "depth": depth,
            "start": start,
            "end": end,
            "row_refs": row_refs[start:end],
            "response": response,
            "parse_success": parsed_ok,
            "empty_response": empty_response,
            "params": params,
            "param_count": len(params),
            "error": error,
            "ancestor_truncated": ancestor_truncated,
            "leaf": False,
            "children": [],
            "truncated": error == "output_truncated",
        }
        nodes.append(node)

        if (
            error == "output_truncated"
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
                left, right
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

    if position != len(responses):
        raise AnalysisError(
            f"{arm_name} raw responses 未全部消费: "
            f"{position}/{len(responses)}"
        )

    return nodes


def build_leaf_by_position(nodes):
    by_position = {}

    for node in nodes:
        if not node["leaf"]:
            continue

        for position in range(
            node["start"], node["end"]
        ):
            by_position[position] = node

    return by_position


def flat_candidates_for_row(
    gold_records,
    row_index,
):
    return [
        item for item in gold_records
        if item.get("table_id") == 9
        and item.get("primary_row_index") == row_index
    ]


def map_raw_parameter(
    arm_name,
    raw_param,
    gold_records,
    gold_by_key,
):
    evidence = raw_param.get("evidence") or {}
    value_type = raw_param.get("value_type")

    if arm_name == "logical":
        row_id = evidence.get("logical_row_id")
        key = (row_id, value_type)

        if key in gold_by_key:
            return key, "logical_row_id"

        return None, "unmapped"

    row_index = evidence.get("row_index")
    candidates = flat_candidates_for_row(
        gold_records, row_index
    )
    candidates = [
        item for item in candidates
        if item.get("value_type") == value_type
    ]

    if len(candidates) == 1:
        return gold_key(candidates[0]), "flat_row"

    raw_value = to_decimal(raw_param.get("value"))
    raw_unit = normalize_unit(raw_param.get("unit"))
    raw_conditions = normalize_conditions(
        raw_param.get("test_conditions")
    )

    exact = []

    for item in candidates:
        if (
            to_decimal(item.get("value")) == raw_value
            and normalize_unit(item.get("unit"))
            == raw_unit
            and normalize_conditions(
                item.get("test_conditions_row_level")
            )
            == raw_conditions
        ):
            exact.append(item)

    if len(exact) == 1:
        return gold_key(exact[0]), "flat_row_value"

    return None, "ambiguous_flat_row"


def build_tracking(
    arm,
    arm_name,
    nodes,
    leaf_by_position,
    row_position_by_id,
    gold_records,
    gold_by_key,
    details_by_key,
):
    raw_keys_by_node = {}

    for node in nodes:
        if not node["leaf"]:
            continue

        raw_key_map = {}

        for raw_index, raw_param in enumerate(
            node["params"]
        ):
            key, method = map_raw_parameter(
                arm_name,
                raw_param,
                gold_records,
                gold_by_key,
            )

            if key is not None:
                raw_key_map[key] = {
                    "raw_index": raw_index,
                    "method": method,
                    "raw_param": raw_param,
                }

        raw_keys_by_node[node["node_id"]] = (
            raw_key_map
        )

    tracking = []

    for gold in gold_records:
        key = gold_key(gold)
        row_id = gold["logical_row_id"]
        position = row_position_by_id.get(row_id)

        if arm_name == "flat":
            position = gold.get("primary_row_index")

        node = (
            leaf_by_position.get(position)
            if position is not None
            else None
        )
        detail = details_by_key.get(key)
        raw_entry = (
            raw_keys_by_node.get(
                node["node_id"], {}
            ).get(key)
            if node is not None
            else None
        )
        raw_output = raw_entry is not None
        parse_success = bool(
            node and node["parse_success"]
        )
        empty_response = bool(
            node and node["empty_response"]
        )

        if detail is None or detail["match_status"] != "matched":
            source_mapping_accepted = None
        elif arm_name == "logical":
            predicted_index = detail["predicted"][
                "index"
            ]
            final_param = arm["extraction"][
                "parameters"
            ][predicted_index]
            status = final_param.get(
                "source_mapping_status"
            )
            source_mapping_accepted = status in {
                "mapped",
                "mapped_corrected",
            }
        else:
            source_mapping_accepted = True

        strict_match = bool(
            detail and detail["strict_match"]
        )
        failure_reasons = []

        if detail is not None and detail[
            "match_status"
        ] == "matched" and not strict_match:
            fields = detail["fields"]

            if not fields["identity"]:
                failure_reasons.append("参数身份错误")

            if not fields["value"] or not fields[
                "unit"
            ]:
                failure_reasons.append(
                    "单位或数值错误"
                )

            if not fields["test_conditions"]:
                failure_reasons.append(
                    "测试条件缺失"
                )

        if (
            detail is None
            or detail["match_status"] == "missing"
        ):
            if node is None:
                failure_reasons.append("未生成")
            elif node["error"] == "output_truncated":
                failure_reasons.append("输出截断")
            elif empty_response:
                failure_reasons.append("空响应")
            elif not parse_success:
                failure_reasons.append("解析失败")
            elif raw_output and source_mapping_accepted is False:
                failure_reasons.append("来源映射失败")
            elif raw_output:
                failure_reasons.append("来源映射失败")
            else:
                failure_reasons.append("未生成")

        primary = (
            failure_reasons[0]
            if failure_reasons else None
        )

        if (
            arm_name == "logical"
            and raw_entry is not None
            and detail is not None
        ):
            raw_source_correct = (
                raw_entry["raw_param"]
                .get("evidence", {})
                .get("logical_row_id")
                == gold["logical_row_id"]
            )
        elif (
            arm_name == "flat"
            and raw_entry is not None
            and detail is not None
        ):
            raw_source_correct = (
                raw_entry["raw_param"]
                .get("evidence", {})
                .get("row_index")
                == gold.get("primary_row_index")
            )
        else:
            raw_source_correct = None

        tracking.append({
            "arm": arm_name,
            "logical_row_id": row_id,
            "value_type": gold["value_type"],
            "parameter_name": gold.get(
                "parameter_name"
            ),
            "chunk": (
                node["node_id"]
                if node is not None else None
            ),
            "chunk_request_index": (
                node["request_index"]
                if node is not None else None
            ),
            "chunk_row_start": (
                node["start"] if node is not None else None
            ),
            "chunk_row_end": (
                node["end"] if node is not None else None
            ),
            "chunk_row_refs": (
                list(node["row_refs"])
                if node is not None else []
            ),
            "chunk_ancestor_truncated": bool(
                node and node["ancestor_truncated"]
            ),
            "input_represented": position is not None,
            "input_representation": (
                "logical_row"
                if arm_name == "logical"
                else (
                    "merged_flat_row"
                    if len({
                        item["logical_row_id"]
                        for item in flat_candidates_for_row(
                            gold_records,
                            position,
                        )
                    }) > 1
                    else "flat_row"
                )
            ),
            "raw_output": raw_output,
            "raw_output_method": (
                raw_entry["method"]
                if raw_entry is not None
                else None
            ),
            "raw_evidence": (
                raw_entry["raw_param"].get(
                    "evidence"
                )
                if raw_entry is not None else None
            ),
            "parse_success": parse_success,
            "empty_response": empty_response,
            "source_mapping_accepted": (
                source_mapping_accepted
            ),
            "strict_match": strict_match,
            "failure_reason": primary,
            "failure_reasons": failure_reasons,
            "raw_source_correct": (
                raw_source_correct
            ),
            "final_fields": (
                detail.get("fields")
                if detail is not None else None
            ),
        })

    return tracking, raw_keys_by_node


def summarize_arm(
    arm,
    tracking,
    nodes,
    raw_keys_by_node,
):
    evaluation = arm["evaluation"]
    reasons = Counter(
        item["failure_reason"]
        for item in tracking
        if item["failure_reason"]
    )
    field_errors = Counter()

    for item in tracking:
        if item["failure_reason"]:
            for reason in item["failure_reasons"]:
                field_errors[reason] += 1

    key_missing = sum(
        1 for item in tracking
        if not item["strict_match"]
        and item["failure_reason"] in {
            "未生成",
            "输出截断",
            "空响应",
            "解析失败",
            "来源映射失败",
        }
    )
    matched_field_error = sum(
        1 for item in tracking
        if item["failure_reason"]
        in {
            "参数身份错误",
            "测试条件缺失",
            "单位或数值错误",
        }
    )

    matched = [
        item for item in tracking
        if item["final_fields"] is not None
    ]
    raw_source_correct = sum(
        1 for item in matched
        if item["raw_source_correct"] is True
    )
    canonical_source_correct = sum(
        1 for item in matched
        if item["source_mapping_accepted"] is True
    )

    metrics = {
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1": evaluation["f1"],
        "key_precision": evaluation["key_precision"],
        "key_recall": evaluation["key_recall"],
        "key_f1": evaluation["key_f1"],
    }

    return {
        **metrics,
        "gold_count": len(tracking),
        "input_represented": sum(
            1 for item in tracking
            if item["input_represented"]
        ),
        "raw_output_count": sum(
            1 for item in tracking
            if item["raw_output"]
        ),
        "parse_success_count": sum(
            1 for item in tracking
            if item["parse_success"]
        ),
        "source_mapping_accepted_count": sum(
            1 for item in tracking
            if item["source_mapping_accepted"] is True
        ),
        "strict_match_count": sum(
            1 for item in tracking
            if item["strict_match"]
        ),
        "matched_key_count": evaluation["matched_keys"],
        "key_missing_count": evaluation[
            "gold_count"
        ] - evaluation["matched_keys"],
        "key_matched_field_error_count": (
            evaluation["matched_keys"]
            - evaluation["strict_matched"]
        ),
        "true_extra_predictions": evaluation[
            "false_positive_count"
        ],
        "duplicate_predictions": evaluation[
            "duplicate_count"
        ],
        "failure_reason_primary": dict(reasons),
        "failure_reason_all": dict(field_errors),
        "raw_source_correct_matched": (
            raw_source_correct
        ),
        "raw_source_accuracy_matched": (
            round(
                raw_source_correct / len(matched),
                4,
            )
            if matched else None
        ),
        "canonical_source_accuracy_matched": (
            round(
                canonical_source_correct / len(matched),
                4,
            )
            if matched else None
        ),
        "chunk_truncated_count": sum(
            1 for node in nodes
            if node["error"] == "output_truncated"
        ),
        "chunk_empty_response_count": sum(
            1 for node in nodes
            if node["leaf"] and node["empty_response"]
        ),
        "chunk_parse_failure_count": sum(
            1 for node in nodes
            if not node["parse_success"]
        ),
        "metrics": metrics,
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
                "row_refs": list(node["row_refs"]),
                "leaf": node["leaf"],
                "truncated": node["truncated"],
                "error": node["error"],
                "parse_success": node["parse_success"],
                "empty_response": node[
                    "empty_response"
                ],
                "param_count": node["param_count"],
                "raw_output_keys": sorted(
                    f"{key[0]}:{key[1]}"
                    for key in raw_keys_by_node.get(
                        node["node_id"], {}
                    )
                ),
                "ancestor_truncated": node[
                    "ancestor_truncated"
                ],
            }
            for node in nodes
        ],
    }


def build_markdown(diagnostic):
    lines = [
        "# B3.3 遗漏归因诊断报告",
        "",
        f"实验源：`{diagnostic['source']['ab_result']}`",
        f"生成时间：{diagnostic['generated_at']}",
        "",
        "## 一、指标口径审计",
        "",
        "| 指标 | Flat | Logical v2 |",
        "|---|---:|---:|",
    ]

    for key in (
        "precision",
        "recall",
        "f1",
        "key_precision",
        "key_recall",
        "key_f1",
        "matched_key_count",
        "strict_match_count",
        "key_missing_count",
        "key_matched_field_error_count",
        "true_extra_predictions",
        "duplicate_predictions",
    ):
        lines.append(
            f"| {key} | "
            f"{diagnostic['summary']['flat'].get(key)} | "
            f"{diagnostic['summary']['logical'].get(key)} |"
        )

    lines += [
        "",
        "定义：",
        "",
        "- 严格 Precision = 严格匹配数 / 原始预测参数数",
        "- 严格 Recall = 严格匹配数 / 51",
        "- Key Precision = 键匹配数 / 原始预测参数数",
        "- Key Recall = 键匹配数 / 51",
        "- 键匹配但字段错误 = 键匹配数 - 严格匹配数",
        "- 真正多余预测 = 不在 Gold 键中的预测；重复键单独统计",
        "",
        "## 二、Chunk 树与失败阶段",
        "",
    ]

    for arm_name in ("flat", "logical"):
        lines += [
            f"### {arm_name}",
            "",
            "| request | parent | rows | leaf | status | params | raw keys |",
            "|---:|---:|---|---|---|---:|---|",
        ]

        for chunk in diagnostic["chunks"][arm_name]:
            status = (
                chunk["error"] or "stop"
            )
            lines.append(
                f"| {chunk['request_index']} | "
                f"{chunk['parent_request_index']} | "
                f"{chunk['row_start']}–{chunk['row_end']} | "
                f"{chunk['leaf']} | {status} | "
                f"{chunk['param_count']} | "
                f"{len(chunk['raw_output_keys'])} |"
            )

        lines.append("")

    lines += [
        "Chunk 级事件：",
        "",
        f"- Flat：截断父 chunk "
        f"{diagnostic['summary']['flat']['chunk_truncated_count']}，"
        f"空叶子响应 "
        f"{diagnostic['summary']['flat']['chunk_empty_response_count']}，"
        f"解析失败 "
        f"{diagnostic['summary']['flat']['chunk_parse_failure_count']}",
        f"- Logical：截断父 chunk "
        f"{diagnostic['summary']['logical']['chunk_truncated_count']}，"
        f"空叶子响应 "
        f"{diagnostic['summary']['logical']['chunk_empty_response_count']}，"
        f"解析失败 "
        f"{diagnostic['summary']['logical']['chunk_parse_failure_count']}",
        "",
        "## 三、失败原因分类",
        "",
        "| 原因 | Flat | Logical v2 |",
        "|---|---:|---:|",
    ]

    for reason in FAILURE_CATEGORIES:
        lines.append(
            f"| {reason} | "
            f"{diagnostic['summary']['flat']['failure_reason_all'].get(reason, 0)} | "
            f"{diagnostic['summary']['logical']['failure_reason_all'].get(reason, 0)} |"
        )

    lines += [
        "",
        "> 注意：chunk 级截断事件数与 Gold 级主失败原因不同。",
        "> 一个被截断的父 chunk 会二分出叶子 chunk；Gold 记录最终归属叶子。",
        "",
        "## 四、专项解释",
        "",
        "### UVLO 为什么 0/7",
        "",
    ]

    for arm_name in ("flat", "logical"):
        uvlo = [
            item for item in diagnostic["tracking"][
                arm_name
            ]
            if item["logical_row_id"] in {
                "t9.p34", "t9.p35", "t9.p36"
            }
        ]
        lines.append(
            f"- {arm_name}: "
            + ", ".join(
                f"{item['logical_row_id']}/{item['value_type']}"
                f"→{item['chunk']}→{item['failure_reason'] or 'OK'}"
                for item in uvlo
            )
        )

    lines += [
        "",
        "### Logical 为什么只有 26 个参数键",
        "",
        "- Logical 的 36 行被分成 4 个基础 chunk。",
        "- 第 1 个基础 chunk 截断后二分，右侧仍输出 p1-p4/p8-p9。",
        "- 第 2 个基础 chunk 截断后二分，左半返回 0 个参数（空响应），右半只输出 p18-p20。",
        "- 最后一个基础 chunk 只输出 p31、p32，未输出 p33 与 UVLO。",
        "",
        "### VDO 为什么 14/14",
        "",
        "- VDO 行集中在 Logical 的第 3、4 个基础 chunk。",
        "- 这些 chunk 未发生截断或空响应。",
        "- Logical 输入携带完整 TJ 与 VOUT 区间，模型原样复制，14 条全部严格匹配。",
        "",
        "### Flat 的 16 条键匹配但严格不匹配",
        "",
    ]

    flat_field_errors = diagnostic["summary"]["flat"][
        "failure_reason_all"
    ]
    lines.append(
        f"- 参数身份错误：{flat_field_errors.get('参数身份错误', 0)}"
    )
    lines.append(
        f"- 测试条件缺失：{flat_field_errors.get('测试条件缺失', 0)}"
    )
    lines.append(
        f"- 单位或数值错误：{flat_field_errors.get('单位或数值错误', 0)}"
    )
    lines += [
        "- 主要原因是 Flat 行缺少完整 TJ/VIN 条件，以及 ICL 被识别成 output_current。",
        "",
        "## 五、Evidence accuracy 审计",
        "",
        "| 指标 | Flat | Logical v2 |",
        "|---|---:|---:|",
    ]

    for key in (
        "raw_source_correct_matched",
        "raw_source_accuracy_matched",
        "canonical_source_accuracy_matched",
        "source_mapping_accepted_count",
    ):
        lines.append(
            f"| {key} | "
            f"{diagnostic['summary']['flat'].get(key)} | "
            f"{diagnostic['summary']['logical'].get(key)} |"
        )

    lines += [
        "",
        "- `accuracy_matched` 的分母是已匹配键，不是全部预测。",
        "- Flat 没有 logical_row_id，最终 Evidence accuracy 依赖 recovered selected row；"
        "不能解释为模型原始来源字段正确。",
        "- Logical 的最终来源字段经过 canonical 重写；"
        "原始来源正确率单独用 raw_source_accuracy_matched 审计。",
        "",
        "## 六、51 条追踪表",
        "",
    ]

    for arm_name in ("flat", "logical"):
        lines += [
            f"### {arm_name}",
            "",
            "| ID | 类型 | Chunk | 进入输入 | 原始输出 | 解析 | 来源接受 | 严格 | 失败原因 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        for item in diagnostic["tracking"][arm_name]:
            lines.append(
                f"| {item['logical_row_id']} | "
                f"{item['value_type']} | "
                f"{item['chunk']} | "
                f"{item['input_represented']} | "
                f"{item['raw_output']} | "
                f"{item['parse_success']} | "
                f"{item['source_mapping_accepted']} | "
                f"{item['strict_match']} | "
                f"{item['failure_reason'] or '—'} |"
            )

        lines.append("")

    lines += [
        "## 七、最小优化方案",
        "",
        "1. 下一轮两臂统一将 `rows_per_chunk` 从 10 降到 5，优先消除截断和空响应。",
        "2. Prompt 显式加入 UVLO threshold / UVLO hysteresis。",
        "3. Prompt 强制复制完整行级测试条件，禁止省略 TJ 或添加参数名前缀。",
        "4. 明确 ICL/ISC 身份为 output_current_limit / short_circuit_current_limit。",
        "5. Logical 输入可减少非必要来源列，降低 Prompt Token 和截断概率。",
        "6. 不进入 B4，先用同一 Gold v2 复测。",
        "",
    ]

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="B3.3 遗漏归因分析"
    )
    parser.add_argument(
        "--result", default=str(DEFAULT_RESULT)
    )
    parser.add_argument(
        "--output-dir", default=None
    )
    args = parser.parse_args(argv)

    result_path = Path(args.result)
    result = load_json(result_path)

    if args.output_dir is None:
        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DEFAULT_OUTPUT_ROOT / stamp
    else:
        output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise AnalysisError(
            f"拒绝覆盖已有目录: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    gold = result["gold_standard"]
    gold_records = gold["expected_parameters"]
    v2_path = Path(
        gold["source"]["logical_rows_v2_path"]
    )
    logical_rows = load_json(v2_path)
    gold_v2 = load_json(
        gold["source"]["gold_v2_path"]
    )
    raw_structure = load_json(
        gold_v2["source"]["raw_structure_path"]
    )
    adapted = adapt_logical_rows(
        logical_rows,
        raw_structure=raw_structure,
        table_ids=[9],
    )
    primary_by_id = {
        record["logical_row_id"]: record[
            "primary_row_index"
        ]
        for record in adapted["tables"][0]["records"]
    }

    # 旧 44 条 Gold 未显式保存 primary_row_index，
    # 分析时从 Logical v2 补齐；不修改源文件。
    for item in gold_records:
        if item.get("primary_row_index") is None:
            item["primary_row_index"] = primary_by_id.get(
                item["logical_row_id"]
            )

    gold_by_key = {
        gold_key(item): item
        for item in gold_records
    }
    flat_source = load_json(
        result["plan"]["target"]["flat_table_source"]
    )

    if "rows" in flat_source:
        flat_table = flat_source
    else:
        flat_table = next(
            table
            for table in flat_source.get("tables", [])
            if table.get("table_id")
            == result["plan"]["target"]["table_id"]
        )

    rows_per_chunk = result["arms"][0][
        "extractor_config"
    ]["rows_per_chunk"]
    max_depth = result["arms"][0][
        "extractor_config"
    ]["max_split_depth"]

    logical_refs = [
        row["logical_row_id"]
        for row in logical_rows["logical_rows"]
    ]
    flat_refs = list(range(len(flat_table["rows"])))

    diagnostic = {
        "schema": "b33_diagnostic/v1",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "ab_result": str(result_path),
            "gold_v2": gold["source"].get(
                "gold_v2_path"
            ),
            "logical_rows_v2": str(v2_path),
            "flat_table": str(
                Path(
                    result["plan"]["target"][
                        "flat_table_source"
                    ]
                )
            ),
        },
        "tracking": {},
        "summary": {},
        "chunks": {},
    }

    for arm in result["arms"]:
        arm_name = arm["arm"]
        row_refs = (
            logical_refs
            if arm_name == "logical"
            else flat_refs
        )
        nodes = build_chunk_tree(
            arm_name,
            arm["raw_llm_responses"],
            row_refs,
            rows_per_chunk,
            max_depth,
        )
        leaf_by_position = build_leaf_by_position(
            nodes
        )
        details_by_key = {
            gold_key(item): item
            for item in arm["evaluation"]["gold_details"]
        }
        row_position_by_id = {
            row_id: index
            for index, row_id in enumerate(
                logical_refs
            )
        }
        tracking, raw_keys_by_node = build_tracking(
            arm,
            arm_name,
            nodes,
            leaf_by_position,
            row_position_by_id,
            gold_records,
            gold_by_key,
            details_by_key,
        )
        diagnostic["tracking"][arm_name] = tracking
        diagnostic["summary"][arm_name] = summarize_arm(
            arm,
            tracking,
            nodes,
            raw_keys_by_node,
        )
        diagnostic["chunks"][arm_name] = (
            diagnostic["summary"][arm_name]["chunks"]
        )

    report = build_markdown(diagnostic)
    json_path = write_json(
        output_dir / "b33_diagnostic.json",
        diagnostic,
    )
    report_path = output_dir / "b33_report.md"
    report_path.write_text(report, encoding="utf-8")

    csv_path = output_dir / "b33_tracking.csv"
    fields = [
        "arm",
        "logical_row_id",
        "value_type",
        "parameter_name",
        "chunk",
        "chunk_row_start",
        "chunk_row_end",
        "input_represented",
        "input_representation",
        "raw_output",
        "raw_output_method",
        "parse_success",
        "empty_response",
        "source_mapping_accepted",
        "strict_match",
        "failure_reason",
        "failure_reasons",
        "chunk_ancestor_truncated",
        "raw_source_correct",
    ]

    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for arm_name in ("flat", "logical"):
            for item in diagnostic["tracking"][
                arm_name
            ]:
                row = {
                    key: item.get(key)
                    for key in fields
                }
                row["failure_reasons"] = ";".join(
                    item.get("failure_reasons") or []
                )
                writer.writerow(row)

    print("=" * 64)
    print("B3.3 诊断已生成")
    print("=" * 64)
    print("目录:", output_dir)
    print("报告:", report_path)
    print("JSON:", json_path)
    print("CSV:", csv_path)

    for arm_name in ("flat", "logical"):
        summary = diagnostic["summary"][arm_name]
        print(
            arm_name,
            "strict=", summary["strict_match_count"],
            "key=", summary["matched_key_count"],
            "raw=", summary["raw_output_count"],
            "failure=", summary["failure_reason_primary"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
