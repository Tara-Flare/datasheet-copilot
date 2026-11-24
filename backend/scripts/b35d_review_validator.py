"""B3.5d：复核 B3.5c 的 13 条 validator_false_reject（只读，不调用 LLM）。

复核目标不是复用 B3.5c 的结论，而是用生产函数重新运行：
- app.extraction.evidence.verify_single_parameter
- app.validation.rules.check_min_typ_max

并记录每条记录为何被判 invalid。仅在可直接证实时给出根因。
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.extraction.evidence import (  # noqa: E402
    find_column,
    normalize_header,
    parse_number,
    verify_single_parameter,
    VALUE_COLUMNS,
)
from app.validation.rules import check_min_typ_max  # noqa: E402
from b35c_score_flat import normalize_gold  # noqa: E402
from b35d_common import (  # noqa: E402
    B35C_ERRORS,
    EXPERIMENT_DIR,
    RUN_ROOT,
    VALIDATOR_DIR,
    load_gold,
    load_json,
    normalize_number,
    normalize_unit_token,
    write_json,
)


REVIEW_SCHEMA = "b35d_validator_review/v1"


ROOT_CAUSE_LABELS = {
    "cross_variant_grouping": (
        "check_min_typ_max 只按 name+test_conditions 分组，"
        "把同型号不同电压档/不同来源表的记录放进同一组，"
        "产生并不存在的 min/typ/max 冲突。"
    ),
    "merged_unit_cell": (
        "来源行单位单元格为合并单元格的续行（空），"
        "verify_single_parameter 用空单位与提取单位比较，判为不一致。"
    ),
    "ambiguous_column_header": (
        "表头带脚注（如 MIN (Note 9)），find_column 归一化后"
        "不等于 min/max，返回 None，证据校验只能 pending_review。"
    ),
    "multi_value_cell": (
        "来源单元格包含多个数值（如 typ='57 63'），"
        "parse_number 只接受单一数字，证据校验只能 pending_review。"
    ),
    "pending_review_collapse": (
        "verify_single_parameter 返回 pending_review 后，"
        "ParameterValidator 只要收到 errors 就写成 invalid，"
        "复核状态被压扁为 hard invalid。"
    ),
}


def selected_tables(sample_id):
    payload = load_json(
        RUN_ROOT / sample_id / "selected_tables.json"
    )

    return {table["table_id"]: table for table in payload["tables"]}


def validation_params(sample_id):
    payload = load_json(
        RUN_ROOT / sample_id / "validation.json"
    )

    return payload["parameters"]


def gold_by_id():
    gold = load_gold()
    index = {}

    for sample in gold["samples"]:
        for record in sample["records"]:
            index[record["gold_id"]] = normalize_gold(record)

    return index


def relation_groups(params):
    grouped = defaultdict(list)

    for index, param in enumerate(params):
        key = (
            str(param.get("name", "")).strip().lower(),
            str(param.get("test_conditions", "")).strip().lower(),
        )
        grouped[key].append(index)

    return grouped


def describe_relation_group(params, group_indices, target_index):
    members = []

    for index in group_indices:
        param = params[index]
        evidence = param.get("evidence", {})
        members.append(
            {
                "index": index,
                "value": param.get("value"),
                "unit": param.get("unit"),
                "value_type": param.get("value_type"),
                "test_conditions": param.get(
                    "test_conditions", ""
                ),
                "table_id": evidence.get("table_id"),
                "row_index": evidence.get("row_index"),
                "page": evidence.get("page"),
                "is_target": index == target_index,
            }
        )

    variants = sorted(
        {
            (member["table_id"], member["row_index"])
            for member in members
        }
    )

    return {
        "member_count": len(members),
        "distinct_sources": len(variants),
        "sources": [
            {"table_id": table_id, "row_index": row_index}
            for table_id, row_index in variants
        ],
        "members": members,
    }


def column_probe(columns, value_type):
    aliases = VALUE_COLUMNS.get(value_type) or set()
    matches = []

    for index, column in enumerate(columns):
        header = normalize_header(column)

        if header in aliases:
            matches.append(
                {
                    "index": index,
                    "column": column,
                    "normalized": header,
                }
            )

    resolved = find_column(columns, aliases)

    return {
        "value_type": value_type,
        "aliases": sorted(aliases),
        "matches": matches,
        "resolved_index": resolved,
        "resolved": resolved is not None,
    }


def unit_probe(columns):
    matches = []

    for index, column in enumerate(columns):
        header = normalize_header(column)

        if header in {"unit", "units"}:
            matches.append(
                {
                    "index": index,
                    "column": column,
                    "normalized": header,
                }
            )

    return matches


def classify(param, verify_status, verify_issues, rule_errors, table, group):
    reasons = []

    if rule_errors:
        reasons.append("cross_variant_grouping")

    if verify_status == "pending_review":
        reasons.append("pending_review_collapse")

    if verify_issues:
        joined = " | ".join(verify_issues)

        if "无法唯一识别参数值对应的表格列" in joined:
            reasons.append("ambiguous_column_header")

        if "来源单元格不是可直接比较的单一数字" in joined:
            reasons.append("multi_value_cell")

        if "与来源单位" in joined and "不一致" in joined:
            reasons.append("merged_unit_cell")

    # 主因：规则冲突优先，其次才是证据校验。
    if rule_errors:
        primary = "cross_variant_grouping"
    elif "ambiguous_column_header" in reasons:
        primary = "ambiguous_column_header"
    elif "multi_value_cell" in reasons:
        primary = "multi_value_cell"
    elif "merged_unit_cell" in reasons:
        primary = "merged_unit_cell"
    elif "pending_review_collapse" in reasons:
        primary = "pending_review_collapse"
    else:
        primary = "unknown"

    return primary, reasons


def build_review():
    errors = [
        error
        for error in load_json(B35C_ERRORS)
        if error["kind"] == "validator"
        and error["category"] == "validator_false_reject"
    ]
    gold_index = gold_by_id()
    cache = {}
    records = []

    for error in errors:
        sample_id = error["sample_id"]

        if sample_id not in cache:
            tables = selected_tables(sample_id)
            params = validation_params(sample_id)
            cache[sample_id] = {
                "tables": tables,
                "params": params,
                "groups": relation_groups(params),
                "rule_errors": check_min_typ_max(params),
            }

        state = cache[sample_id]
        tables = state["tables"]
        params = state["params"]
        index = error["prediction_index"]
        param = params[index]
        evidence = param.get("evidence", {})
        table = tables.get(evidence.get("table_id"))
        verify_status, verify_issues = verify_single_parameter(
            param, tables
        )
        rule_errors = state["rule_errors"].get(index, [])
        key = (
            str(param.get("name", "")).strip().lower(),
            str(param.get("test_conditions", "")).strip().lower(),
        )
        group = describe_relation_group(
            params, state["groups"][key], index
        )
        columns = table.get("columns", []) if table else []
        rows = table.get("rows", []) if table else []
        row_index = evidence.get("row_index")
        source_row = (
            rows[row_index]
            if isinstance(row_index, int)
            and 0 <= row_index < len(rows)
            else None
        )
        unit_columns = unit_probe(columns)
        unit_cell = None

        if (
            source_row is not None
            and unit_columns
            and unit_columns[0]["index"] < len(source_row)
        ):
            unit_cell = source_row[unit_columns[0]["index"]]

        primary, reasons = classify(
            param,
            verify_status,
            verify_issues,
            rule_errors,
            table,
            group,
        )
        gold = gold_index.get(error["gold_id"])

        records.append(
            {
                "sample_id": sample_id,
                "vendor": error["vendor"],
                "gold_id": error["gold_id"],
                "prediction_index": index,
                "gold_identity": error["gold_identity"],
                "gold_value": error["gold_value"],
                "gold_unit": error["gold_unit"],
                "gold_value_type": error["gold_value_type"],
                "gold_conditions": error["gold_conditions"],
                "gold_source": {
                    "table_id": gold["table_id"] if gold else None,
                    "row_index": gold["row_index"] if gold else None,
                    "page": gold["page"] if gold else None,
                },
                "prediction": {
                    "name": param.get("name"),
                    "value": param.get("value"),
                    "unit": param.get("unit"),
                    "value_type": param.get("value_type"),
                    "test_conditions": param.get(
                        "test_conditions", ""
                    ),
                    "table_id": evidence.get("table_id"),
                    "row_index": evidence.get("row_index"),
                    "page": evidence.get("page"),
                    "source_text": evidence.get("source_text", ""),
                    "validation_status": param.get(
                        "validation_status"
                    ),
                    "validation_errors": param.get(
                        "validation_errors", []
                    ),
                },
                "reproduced": {
                    "verify_status": verify_status,
                    "verify_issues": verify_issues,
                    "rule_errors": rule_errors,
                    "final_validation_status": param.get(
                        "validation_status"
                    ),
                    "final_validation_errors": param.get(
                        "validation_errors", []
                    ),
                    "matches_b35c": (
                        param.get("validation_errors", [])
                        == (
                            [error["validation_errors"]]
                            if error["validation_errors"]
                            else []
                        )
                    ),
                },
                "root_cause": primary,
                "root_cause_flags": reasons,
                "root_cause_note": ROOT_CAUSE_LABELS.get(
                    primary, ""
                ),
                "relation_group": group,
                "table_probe": {
                    "columns": columns,
                    "row_count": len(rows),
                    "source_row": source_row,
                    "min_column": column_probe(columns, "min"),
                    "typ_column": column_probe(columns, "typ"),
                    "max_column": column_probe(columns, "max"),
                    "unit_columns": unit_columns,
                    "source_unit_cell": unit_cell,
                    "source_unit_blank": not str(
                        unit_cell or ""
                    ).strip(),
                },
                "content_matches_gold": (
                    normalize_unit_token(param.get("unit"))
                    == normalize_unit_token(error["gold_unit"])
                    and normalize_number(param.get("value"))
                    == normalize_number(error["gold_value"])
                    and param.get("value_type")
                    == error["gold_value_type"]
                ),
            }
        )

    counts = Counter(record["root_cause"] for record in records)
    per_sample = defaultdict(Counter)

    for record in records:
        per_sample[record["sample_id"]][
            record["root_cause"]
        ] += 1

    return {
        "schema": REVIEW_SCHEMA,
        "source": "B3.5c flat_baseline_r1 metrics.errors.json",
        "note": (
            "复核只重新运行生产证据校验与 min/typ/max 规则，"
            "不调用 LLM，不修改生产代码。"
        ),
        "total": len(records),
        "root_cause_counts": dict(counts),
        "per_sample_root_cause_counts": {
            key: dict(value)
            for key, value in per_sample.items()
        },
        "all_reproduced": all(
            record["reproduced"]["final_validation_status"]
            == "invalid"
            for record in records
        ),
        "records": records,
    }


def write_csv(records, path):
    fieldnames = [
        "sample_id",
        "gold_id",
        "prediction_index",
        "gold_identity",
        "gold_value",
        "gold_unit",
        "gold_value_type",
        "gold_conditions",
        "prediction_value",
        "prediction_unit",
        "prediction_value_type",
        "prediction_conditions",
        "prediction_table_id",
        "prediction_row_index",
        "final_validation_status",
        "final_validation_errors",
        "verify_status",
        "verify_issues",
        "rule_errors",
        "root_cause",
        "root_cause_flags",
        "distinct_sources_in_group",
        "group_member_count",
        "source_unit_cell",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "gold_id": record["gold_id"],
                    "prediction_index": record[
                        "prediction_index"
                    ],
                    "gold_identity": record["gold_identity"],
                    "gold_value": record["gold_value"],
                    "gold_unit": record["gold_unit"],
                    "gold_value_type": record[
                        "gold_value_type"
                    ],
                    "gold_conditions": record["gold_conditions"],
                    "prediction_value": record["prediction"][
                        "value"
                    ],
                    "prediction_unit": record["prediction"][
                        "unit"
                    ],
                    "prediction_value_type": record[
                        "prediction"
                    ]["value_type"],
                    "prediction_conditions": record[
                        "prediction"
                    ]["test_conditions"],
                    "prediction_table_id": record["prediction"][
                        "table_id"
                    ],
                    "prediction_row_index": record[
                        "prediction"
                    ]["row_index"],
                    "final_validation_status": record[
                        "prediction"
                    ]["validation_status"],
                    "final_validation_errors": " | ".join(
                        record["prediction"][
                            "validation_errors"
                        ]
                    ),
                    "verify_status": record["reproduced"][
                        "verify_status"
                    ],
                    "verify_issues": " | ".join(
                        record["reproduced"]["verify_issues"]
                    ),
                    "rule_errors": " | ".join(
                        record["reproduced"]["rule_errors"]
                    ),
                    "root_cause": record["root_cause"],
                    "root_cause_flags": " | ".join(
                        record["root_cause_flags"]
                    ),
                    "distinct_sources_in_group": record[
                        "relation_group"
                    ]["distinct_sources"],
                    "group_member_count": record[
                        "relation_group"
                    ]["member_count"],
                    "source_unit_cell": record["table_probe"][
                        "source_unit_cell"
                    ],
                }
            )


def markdown(review):
    lines = [
        "# B3.5d Validator 疑似误拒复核",
        "",
        "本报告只重新运行生产函数，不调用 LLM，不修改生产代码。",
        "",
        f"- 复核记录：{review['total']}",
        f"- 全部可复现为 `invalid`：{review['all_reproduced']}",
        "",
        "## 根因分布",
        "",
        "| 根因 | 数量 | 说明 |",
        "| --- | ---: | --- |",
    ]

    for cause, count in sorted(
        review["root_cause_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(
            f"| `{cause}` | {count} | "
            f"{ROOT_CAUSE_LABELS.get(cause, '')} |"
        )

    lines.extend(
        [
            "",
            "## 逐条复核",
            "",
            "| Gold | 预测 | 最终错误 | 复核错误 | 根因 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for record in review["records"]:
        prediction = record["prediction"]
        lines.append(
            "| {gold} | {name}={value} {unit} {vt} | {final} | "
            "{verify} | `{cause}` |".format(
                gold=record["gold_id"],
                name=prediction["name"],
                value=prediction["value"],
                unit=prediction["unit"],
                vt=prediction["value_type"],
                final="; ".join(
                    prediction["validation_errors"]
                )
                or "-",
                verify="; ".join(
                    record["reproduced"]["verify_issues"]
                    + record["reproduced"]["rule_errors"]
                )
                or "-",
                cause=record["root_cause"],
            )
        )

    lines.extend(
        [
            "",
            "## 关键证据",
            "",
            "### 1. 跨电压档 min/typ/max 误比较",
            "",
            "`check_min_typ_max` 的分组键只有 "
            "`(name, test_conditions)`。AP2112/NCP170 的多个"
            "同型号电压档来自不同来源行或不同表，"
            "条件文本相同，于是被放进同一组。",
            "",
        ]
    )

    for record in review["records"]:
        if record["root_cause"] != "cross_variant_grouping":
            continue

        group = record["relation_group"]
        lines.append(
            f"- `{record['gold_id']}`：组内 {group['member_count']} 条，"
            f"来自 {group['distinct_sources']} 个来源"
            f"（table_id, row_index）：{group['sources']}"
        )

    lines.extend(
        [
            "",
            "### 2. 合并单元格单位续行为空",
            "",
        ]
    )

    for record in review["records"]:
        if "merged_unit_cell" not in record[
            "root_cause_flags"
        ]:
            continue

        lines.append(
            f"- `{record['gold_id']}`：来源行 "
            f"{record['prediction']['row_index']} 的单位单元格为 "
            f"{record['table_probe']['source_unit_cell']!r}，"
            f"提取单位为 {record['prediction']['unit']!r}。"
        )

    lines.extend(
        [
            "",
            "### 3. 表头脚注导致列无法识别",
            "",
        ]
    )

    for record in review["records"]:
        if record["root_cause"] != "ambiguous_column_header":
            continue

        probe = record["table_probe"]
        detail = (
            probe["max_column"]
            if record["gold_value_type"] == "max"
            else probe["min_column"]
        )
        lines.append(
            f"- `{record['gold_id']}`：表头 {detail['matches']}，"
            f"`find_column` 返回 {detail['resolved_index']}。"
        )

    lines.extend(
        [
            "",
            "### 4. 多值单元格只能 pending_review",
            "",
        ]
    )

    for record in review["records"]:
        if record["root_cause"] != "multi_value_cell":
            continue

        lines.append(
            f"- `{record['gold_id']}`："
            f"{record['reproduced']['verify_issues']}；"
            f"来源行 {record['table_probe']['source_row']}。"
        )

    lines.extend(
        [
            "",
            "## 修复边界",
            "",
            "- `check_min_typ_max` 必须把来源表/来源行或电压档纳入分组，"
            "否则只能关掉跨组比较；建议只比较同一来源行的 "
            "min/typ/max。",
            "- 合并单元格单位应回退到同一逻辑行/表级单位，"
            "不能把空单位当真实单位。",
            "- 表头匹配应先剥离脚注括号，再精确匹配。",
            "- 多值单元格应进入结构恢复拆分，"
            "而不是把 pending_review 压成 invalid。",
            "- 不得为提 Recall 放宽真实的数值/单位错误校验；"
            "上述修复只改变分组与列解析。",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    review = build_review()
    write_json(VALIDATOR_DIR / "review.json", review)
    write_csv(
        review["records"], VALIDATOR_DIR / "review.csv"
    )
    report_path = (
        EXPERIMENT_DIR / "reports/validator_review.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        markdown(review), encoding="utf-8"
    )

    print("复核记录:", review["total"])
    print("根因:", review["root_cause_counts"])
    print("全部复现:", review["all_reproduced"])
    print("输出:", VALIDATOR_DIR / "review.json")
    print("报告:", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
