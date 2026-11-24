"""B3.5d：根据已生成的离线产物输出专项报告与总报告。

只读消费 trace/structure/cost/validator 的 JSON，不调用 LLM。
所有“结构可恢复率”均明确标注为非真实 F1。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b35d_common import (  # noqa: E402
    EXPERIMENT_DIR,
    RAW_DIR,
    RUN_ROOT,
    TRACE_DIR,
    VALIDATOR_DIR,
    attach_events,
    load_gold,
    load_json,
    write_json,
)


COST_PATH = EXPERIMENT_DIR / "cost/cost_breakdown.json"
STRUCTURE_PATH = (
    EXPERIMENT_DIR / "structure/structure_compare.json"
)
REPORT_DIR = EXPERIMENT_DIR / "reports"


def load_optional(path):
    return load_json(path) if Path(path).is_file() else None


def selected_tables(sample_id):
    payload = load_json(
        RUN_ROOT / sample_id / "selected_tables.json"
    )

    return {
        table["table_id"]: table for table in payload["tables"]
    }


def validation_params(sample_id):
    return load_json(
        RUN_ROOT / sample_id / "validation.json"
    )["parameters"]


def table_call_facts(sample_id, table_id):
    events = attach_events(sample_id)
    calls = [
        event
        for event in events
        if event["kind"] == "extract"
        and event["table_id"] == table_id
    ]

    return {
        "calls": len(calls),
        "finish_reasons": dict(
            Counter(event["finish"] for event in calls)
        ),
        "parameter_counts": [
            len(event["parameters"]) for event in calls
        ],
        "content_lengths": [
            event["content_length"] for event in calls
        ],
        "prompt_tokens": sum(
            event["prompt_tokens"] for event in calls
        ),
        "completion_tokens": sum(
            event["completion_tokens"] for event in calls
        ),
        "reasoning_tokens": sum(
            event["reasoning_tokens"] for event in calls
        ),
        "chunks": [
            {
                "index": event["index"],
                "start": event["start"],
                "rows": event["rows"],
                "depth": event["depth"],
                "finish": event["finish"],
                "content_length": event["content_length"],
                "parameters": len(event["parameters"]),
            }
            for event in calls
        ],
    }


def number_tokens(value):
    import re

    return re.findall(r"[-+]?\d+(?:\.\d+)?", str(value or ""))


def adi_facts():
    sample_id = "adi_lt3045"
    structure = load_optional(STRUCTURE_PATH)
    sample = next(
        item
        for item in structure["samples"]
        if item["sample_id"] == sample_id
    )
    gold = load_gold()
    gold_sample = next(
        item
        for item in gold["samples"]
        if item["sample_id"] == sample_id
    )
    tables = selected_tables(sample_id)
    table = tables[4]
    rows = table["rows"]
    columns = table["columns"]
    value_indices = {
        "min": columns.index("MIN"),
        "typ": columns.index("TYP"),
        "max": columns.index("MAX"),
    }
    multi_rows = []

    for index, row in enumerate(rows):
        flags = {}

        for value_type, column in value_indices.items():
            if column < len(row):
                count = len(number_tokens(row[column]))
                flags[value_type] = count

        if any(count > 1 for count in flags.values()):
            multi_rows.append({"row_index": index, **flags})

    gold_records = [
        record
        for record in gold_sample["records"]
        if record["source"]["table_id"] == 4
    ]
    multi_gold = 0

    for record in gold_records:
        column = value_indices[record["value_type"]]
        cell = rows[record["source"]["row_index"]][column]

        if len(number_tokens(cell)) > 1:
            multi_gold += 1

    params = validation_params(sample_id)
    by_table = Counter(
        param["evidence"]["table_id"] for param in params
    )
    logical = load_json(
        EXPERIMENT_DIR
        / "structure/logical/"
        "adi_lt3045.table4.logical_rows.v1.json"
    )
    adapter = load_json(
        EXPERIMENT_DIR
        / "structure/logical/"
        "adi_lt3045.table4.adapter.json"
    )
    adapter_table = adapter["tables"][0]
    retained = [
        {
            "logical_row_id": row["logical_row_id"],
            "min": row["min"],
            "typ": row["typ"],
            "max": row["max"],
            "unit": row["unit"],
        }
        for row in adapter_table["records"]
    ]

    return {
        "sample_id": sample_id,
        "selected_table_ids": sorted(tables),
        "gold_table_id": 4,
        "table4_columns": columns,
        "table4_row_count": len(rows),
        "gold_record_count": len(gold_records),
        "gold_records_in_multi_value_cell": multi_gold,
        "flat_multi_value_rows": multi_rows,
        "flat_multi_value_row_count": len(multi_rows),
        "table4_calls": table_call_facts(sample_id, 4),
        "logical_row_count": len(logical["logical_rows"]),
        "logical_rows_without_values": sum(
            1
            for row in logical["logical_rows"]
            if not any(
                str(row.get(field) or "").strip()
                for field in ("min", "typ", "max")
            )
        ),
        "adapter_rows_total": adapter_table["row_count"],
        "adapter_excluded_rows": len(
            adapter_table.get("excluded_rows", [])
        ),
        "adapter_excluded_reasons": dict(
            Counter(
                row.get("reason")
                for row in adapter_table.get("excluded_rows", [])
            )
        ),
        "adapter_retained_rows": retained,
        "structure_recovered_complete": sample[
            "recovered_rows_complete"
        ],
        "structure_recovered_coverage": sample[
            "recovered_coverage"
        ],
        "adapter_complete": sample["adapter_complete"],
        "adapter_coverage": sample["adapter_coverage"],
        "final_parameters_by_table": {
            str(key): value
            for key, value in sorted(by_table.items())
        },
        "final_parameters_total": len(params),
        "table4_final_parameters": by_table.get(4, 0),
        "gold_table": {
            "rows": rows,
            "columns": columns,
        },
    }


def adi_markdown(facts):
    call = facts["table4_calls"]
    lines = [
        "# B3.5d ADI LT3045 table 4 专项报告",
        "",
        "## 结论",
        "",
        "ADI table 4 的 18 条 Gold 全部为 FN。失败**不是**由输出截断、"
        "TableSelector 漏选或评分匹配造成：",
        "",
        f"- table 4 已被选中：selected_table_ids = "
        f"{facts['selected_table_ids']}；",
        f"- table 4 共 {call['calls']} 次提取调用，"
        f"finish_reason = {call['finish_reasons']}，"
        f"返回参数数 = {call['parameter_counts']}；",
        f"- 两次调用内容长度 = {call['content_lengths']} 字符，"
        "均为合法 JSON 空参数数组，不是 `length` 截断。",
        "",
        "最早可证实的失败在**输入结构**：",
        "",
        f"- Gold table 4 共 {facts['gold_record_count']} 条记录，"
        f"其中 {facts['gold_records_in_multi_value_cell']} 条的 Gold 数值"
        "位于包含多个数字的单元格；",
        f"- Flat 输入有 {facts['flat_multi_value_row_count']} 行"
        f"（row_index = "
        f"{[row['row_index'] for row in facts['flat_multi_value_rows']]}）"
        "的 MIN/TYP/MAX 单元格含多个数值或多个单位；",
        f"- 离线 StructureRecovery 生成 {facts['logical_row_count']} 条"
        f"逻辑行，其中 {facts['logical_rows_without_values']} 条"
        "没有任何 min/typ/max 数值；",
        f"- LogicalTableAdapter 保留 {facts['adapter_rows_total']} 行、"
        f"排除 {facts['adapter_excluded_rows']} 行，"
        f"排除原因 {facts['adapter_excluded_reasons']}；"
        f"Gold 行完整可覆盖 {facts['adapter_complete']}/"
        f"{facts['gold_record_count']}；",
        "- 同时 LLM 对 table 4 直接返回 0 条参数，"
        "因此结构层与 LLM 层都失败，但结构层更早。",
        "",
        "## 证据 1：table 4 的 Flat 输入",
        "",
        "| row | PARAMETER | MIN | TYP | MAX | UNITS |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]

    for index, row in enumerate(facts["gold_table"]["rows"]):
        lines.append(
            "| {row} | {name} | {min} | {typ} | {max} | {unit} |".format(
                row=index,
                name=str(row[0]).replace("|", "\\|"),
                min=row[3],
                typ=row[4],
                max=row[5],
                unit=row[6],
            )
        )

    lines.extend(
        [
            "",
            "可以看到 table 4 的多值单元格不是偶发："
            "`1 10 20`、`570 230`、`710 700 330`、"
            "`150 500 100`、`µA µA µA`、`mA · kΩ mA mA` "
            "都来自 Docling 的合并/压缩单元格。",
            "",
            "## 证据 2：StructureRecovery / Adapter",
            "",
            f"- logical rows: {facts['logical_row_count']}",
            f"- 无任何数值的逻辑行: "
            f"{facts['logical_rows_without_values']}",
            f"- adapter 保留行: {facts['adapter_rows_total']}",
            f"- adapter 排除行: {facts['adapter_excluded_rows']} "
            f"（{facts['adapter_excluded_reasons']}）",
            f"- StructureRecovery 完整覆盖 Gold: "
            f"{facts['structure_recovered_complete']}/"
            f"{facts['gold_record_count']}",
            f"- Adapter 完整覆盖 Gold: {facts['adapter_complete']}/"
            f"{facts['gold_record_count']}",
            "",
            "adapter 保留的 3 行也不是可验证数值：",
            "",
            "| logical_row_id | min | typ | max | unit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in facts["adapter_retained_rows"]:
        lines.append(
            f"| {row['logical_row_id']} | {row['min']!r} | "
            f"{row['typ']!r} | {row['max']!r} | {row['unit']!r} |"
        )

    lines.extend(
        [
            "",
            "## 证据 3：最终预测去了哪里",
            "",
            f"- 最终参数总数: {facts['final_parameters_total']}",
            f"- 按 table_id 分布: "
            f"{facts['final_parameters_by_table']}",
            f"- table 4 的最终参数: "
            f"{facts['table4_final_parameters']}",
            "",
            "Gold table 4 的 in-scope 预测为 0；"
            "24 条来自 table 3、33 条来自 table 11，全部是 out-of-scope。"
            "这说明 ADI 的损失不是“抽到了但没匹配”，"
            "而是 Gold 表本身没有产出可用参数。",
            "",
            "## 成本",
            "",
            f"- table 4 两次调用：prompt {call['prompt_tokens']}，"
            f"completion {call['completion_tokens']}，"
            f"reasoning {call['reasoning_tokens']}。",
            "- 截断集中在 table 3（2 次 length）和 table 11"
            "（1 次 length），与 table 4 的 0 输出无关。",
            "",
            "## 最小修复",
            "",
            "1. `structure_recovery.py` 必须按 MIN/TYP/MAX 列拆分"
            "多值单元格，并保留每个值对应的条件序列；"
            "单位列的多单位必须与数值位置对齐。",
            "2. `logical_table_adapter.py` 不得把 "
            "`20V (Note 11)`、`= 0V` 这类条件碎片当 min 值；"
            "排除原因应区分“缺值”和“值不可解析”。",
            "3. Flat/Logical Prompt 需要显式覆盖 current limit、"
            "programmable current limit、scale factor 等身份，"
            "但这一项不能弥补 table 4 的 0 输出。",
            "",
            "> 以上结构覆盖数字是离线可恢复性，不是真实提取 F1。",
            "",
        ]
    )

    return "\n".join(lines)


def structure_markdown(structure, trace):
    totals = structure["totals"]
    lines = [
        "# B3.5d Flat vs Recovered 结构对照",
        "",
        "> 本报告只描述离线结构可恢复性/可覆盖率。"
        "Recovered/Adapter 未进入 B3.5c 运行，"
        "这里的百分比**不是**真实提取 F1，也不代表 Recall 改善。",
        "",
        "## 全局",
        "",
        "| 指标 | 数量 | 比例 |",
        "| --- | ---: | ---: |",
        f"| Gold 记录 | {totals['gold_records']} | 100% |",
        f"| StructureRecovery 完整行 | "
        f"{totals['recovered_rows_complete']} | "
        f"{totals['recovered_coverage']:.2%} |",
        f"| Adapter 命中来源行 | "
        f"{totals['adapter_rows_matched']} | "
        f"{totals['adapter_rows_matched'] / totals['gold_records']:.2%} |",
        f"| Adapter 身份匹配 | "
        f"{totals['adapter_identity_matched']} | "
        f"{totals['adapter_identity_coverage']:.2%} |",
        f"| Adapter 数值存在 | "
        f"{totals['adapter_value_present']} | "
        f"{totals['adapter_value_coverage']:.2%} |",
        f"| Adapter 单位存在 | "
        f"{totals['adapter_unit_present']} | "
        f"{totals['adapter_unit_present'] / totals['gold_records']:.2%} |",
        f"| Adapter 条件完整 | "
        f"{totals['adapter_conditions_complete']} | "
        f"{totals['adapter_conditions_complete'] / totals['gold_records']:.2%} |",
        f"| Adapter 来源行号正确 | "
        f"{totals['adapter_row_index_correct']} | "
        f"{totals['adapter_row_index_correct'] / totals['gold_records']:.2%} |",
        f"| Adapter 完整可覆盖 | "
        f"{totals['adapter_complete']} | "
        f"{totals['adapter_coverage']:.2%} |",
        "",
        "## 分样本",
        "",
        "| 样本 | Gold | Recovered 完整 | Recovered 覆盖率 | "
        "Adapter 完整 | Adapter 覆盖率 | Adapter 行数 | 排除行 | 排除原因 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for sample in structure["samples"]:
        lines.append(
            "| {sample} | {gold} | {rec} | {rec_rate:.2%} | "
            "{ad} | {ad_rate:.2%} | {rows} | {excluded} | {reasons} |".format(
                sample=sample["sample_id"],
                gold=sample["gold_records"],
                rec=sample["recovered_rows_complete"],
                rec_rate=sample["recovered_coverage"],
                ad=sample["adapter_complete"],
                ad_rate=sample["adapter_coverage"],
                rows=sample["adapter_rows_total"],
                excluded=sample["adapter_excluded_rows"],
                reasons="; ".join(
                    f"{key}:{value}"
                    for key, value in (
                        sample["adapter_excluded_reasons"] or {}
                    ).items()
                )
                or "-",
            )
        )

    lines.extend(
        [
            "",
            "## 关键缺口",
            "",
            "### ADI：多值单元格未拆分",
            "",
            "- Flat table 4 的 6 行含多值/多单位；18 条 Gold 中 17 条"
            "位于多值单元格。",
            "- StructureRecovery 0/18 完整；Adapter 0/18 完整。",
            "- 这是 ADI 18 条 FN 的输入侧最早失败点。",
            "",
            "### AP2112：值/单位恢复好，条件文本受损",
            "",
            "- Recovery 21/21 完整，Adapter 值/单位 21/21。",
            "- Adapter 条件完整仅 14/21：重复 Conditions 列被拼成 "
            "`1mA ≤ IOUT | IOUT ≤ 30mA`，`|` 破坏条件。",
            "- 因此结构指标好并不等于真实 F1 好；AP 的 10 条 FN "
            "主要是 LLM 省略，不是结构缺失。",
            "",
            "### NCP170：µA 被 Docling 解析为 mA",
            "",
            "- Adapter 单位仅 14/19；5 条 Gold 单位 µA 在 Flat/Logical "
            "中为 mA 或 A。",
            "- 2 条 ground current 的 LLM 原始响应忠实复制了错误的 mA，"
            "不能算 LLM 擅自换算。",
            "",
            "### Renesas：Short Circuit 只保留 typ",
            "",
            "- Adapter 完整 8/18；short-circuit 的 min/max 丢失。",
            "- 但 Flat 输入本身是单值，LLM 仍然省略；"
            "所以 Renesas 的 10 条 FN 最早失败在 LLM，"
            "不是结构恢复。",
            "",
            "## 与 FN 归因的交叉验证",
            "",
            "| 样本 | FN | 输入结构 | LLM 提取 | 结构完整覆盖 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    trace_by_sample = defaultdict(Counter)

    for record in trace["records"]:
        trace_by_sample[record["sample_id"]][
            record["stage_group"]
        ] += 1

    fn_counts = Counter(
        record["sample_id"] for record in trace["records"]
    )

    for sample in structure["samples"]:
        sample_id = sample["sample_id"]
        counts = trace_by_sample[sample_id]
        lines.append(
            f"| {sample_id} | {fn_counts[sample_id]} | "
            f"{counts.get('input_structure', 0)} | "
            f"{counts.get('llm_extraction', 0)} | "
            f"{sample['recovered_rows_complete']}/"
            f"{sample['gold_records']} |"
        )

    lines.extend(
        [
            "",
            "结论：结构恢复不是万能修复。"
            "AP/Renesas 的结构覆盖即使提高，也不能替代 Prompt 身份覆盖；"
            "ADI 则必须先解决多值单元格拆分。",
            "",
        ]
    )

    return "\n".join(lines)


def fix_priority_markdown(trace, structure, cost, review):
    stage_counts = trace["stage_group_counts"]
    sample_stage = trace["per_sample_group_counts"]
    totals = cost["totals"]
    llm_by_sample = ", ".join(
        f"{sample_id} {counts.get('llm_extraction', 0)}"
        for sample_id, counts in sample_stage.items()
    )

    lines = [
        "# B3.5d 修复优先级与影响范围",
        "",
        "> 影响范围是“最多可解除的 FN/误拒数量”上界，"
        "不是承诺的 F1 提升。不得把结构指标当作真实 F1。",
        "",
        "## 1. 优先级总表",
        "",
        "| 优先级 | 修复项 | 涉及文件 | 可证实问题 | 影响上界 |",
        "| --- | --- | --- | --- | --- |",
        "| P0 | StructureRecovery 多值单元格拆分 + 单位对齐 | "
        "`backend/app/parsers/structure_recovery.py`, "
        "`backend/app/extraction/logical_table_adapter.py` | "
        "ADI 18 条 Gold 的 17 条位于多值单元格；"
        "Recovery 0/18、Adapter 0/18 | "
        "ADI 18 条输入阻塞；NCP/Renesas 部分数值丢失 |",
        "| P0 | Flat/Logical Prompt 身份覆盖 | "
        "`backend/app/extraction/prompts.py`, "
        "`backend/app/extraction/deepseek_extractor.py` | "
        f"{stage_counts.get('llm_extraction', 0)} 条 FN 为 LLM 省略；"
        f"47/50 条 FN 身份不在 Flat “重点提取”列表 | "
        f"{stage_counts.get('llm_extraction', 0)} 条 FN |",
        "| P1 | Validator 分组/表头/合并单位/状态语义 | "
        "`backend/app/validation/rules.py`, "
        "`backend/app/extraction/evidence.py`, "
        "`backend/app/validation/validator.py` | "
        f"{review['total']} 条内容与 Gold 完全一致却被判 invalid | "
        "13 条误拒（本轮直接 F1 影响 0） |",
        "| P1 | Docling µ 前缀与单位字形保真 | "
        "`backend/app/parsers/structure_recovery.py`（单位归一化） | "
        "NCP 5 条 Gold µA 在输入中为 mA/A | "
        "NCP 最多 5 条 FN |",
        "| P2 | TableSelector / 重复电压档去重 | "
        "`backend/app/extraction/table_selector.py`（本轮只读） | "
        f"非 Gold scope 表提取调用占 "
        f"{totals['non_gold_table_call_ratio']:.2%}；"
        "AP/NCP 的 4 个重复表放大跨档校验冲突 | "
        "成本与误拒，不直接增加 Recall |",
        "| P2 | 分块/max_tokens/retry | "
        "`backend/app/extraction/deepseek_extractor.py` | "
        f"{totals['length_calls']['calls']} 次 length 截断，"
        f"占 completion token "
        f"{totals['length_completion_token_share']:.2%}；"
        f"retry 占调用 {totals['retry_call_ratio']:.2%}，"
        "invalid→validated 纠正 0 | 成本，不直接增加 Recall |",
        "| P3 | 评分匹配 | `backend/scripts/b35c_score_flat.py` | "
        "50 条 FN 中评分类 0 | 0 |",
        "",
        "## 2. 逐项说明",
        "",
        "### P0-1 StructureRecovery 多值单元格拆分",
        "",
        "- 最早失败阶段：`input_structure` "
        f"{stage_counts.get('input_structure', 0)}/50。",
        "- ADI table 4 的 Flat 输入：`1 10 20`、"
        "`570 230`、`710 700 330`、`150 500 100`，"
        "单位列 `µA µA µA`、`mA · kΩ mA mA`。",
        "- 当前 StructureRecovery 未拆；Adapter 排除 12/15 行，"
        "保留的 3 行是条件碎片。",
        "- 最小验收：每个逻辑行只含一个 min/typ/max 数值；"
        "单位与数值位置一一对应；条件按同一索引对齐。",
        "- **边界**：不得把范围值、公式、Note 文本强拆成数字；"
        "无法解析时应保留原文并标记，而不是猜。",
        "",
        "### P0-2 Prompt 身份覆盖",
        "",
        f"- 逐样本 LLM 省略：{llm_by_sample}。",
        "- 26 条省略全部属于 `line_regulation`、`load_regulation`、"
        "`shutdown_current`、`short_circuit_current`、`noise` "
        "等不在 Flat “重点提取”列表的身份。",
        "- 这些记录的来源行是单一数值、调用 `finish_reason=stop`、"
        "原始响应完全没有该身份，因此是提取遗漏，不是截断，"
        "也不是评分。",
        "- 仍需注意：提示词“重点提取”不等于“只能提取”，"
        "因此只能证明覆盖缺口与遗漏相关，不能证明唯一因果。",
        "",
        "### P1-1 Validator 误拒",
        "",
        "复核的 13 条全部内容与 Gold 一致，且在 B3.5c 中已被算作 TP：",
        "",
        "| 根因 | 数量 | 最小修复 |",
        "| --- | ---: | --- |",
    ]

    for cause, count in sorted(
        review["root_cause_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        fixes = {
            "cross_variant_grouping": (
                "`check_min_typ_max` 分组加入来源 "
                "table_id/row_index 或电压档；"
                "只比较同一来源行"
            ),
            "merged_unit_cell": (
                "单位空时回退到同一逻辑行/表级单位，"
                "不得用空字符串比较"
            ),
            "ambiguous_column_header": (
                "`find_column` 先剥离脚注括号再精确匹配"
            ),
            "multi_value_cell": (
                "多值单元格交给结构拆分；"
                "pending_review 不得压成 invalid"
            ),
        }
        lines.append(
            f"| `{cause}` | {count} | "
            f"{fixes.get(cause, '')} |"
        )

    lines.extend(
        [
            "",
            "> 重要：B3.5c 评分不按 `validation_status` 过滤，"
            "所以这 13 条误拒**没有降低本轮 Recall**。"
            "修复它们是生产可信度修复，不是提 Recall 手段。",
            "> 不得为提 Recall 放松数值/单位真实性校验；"
            "上述修复只改分组键、表头解析、"
            "合并单位回退和 pending 状态语义。",
            "",
            "### P1-2 Docling µ 前缀",
            "",
            "- NCP170 的 Gold 备注记录了逐字形证据："
            "PDF 为 µA，Docling raw 为 mA。",
            "- 2 条 ground current 的 LLM 输出是 `0.5 mA`/`0.9 mA`，"
            "与 Flat 输入一致；把责任归给 LLM 会误判。",
            "- 修复必须在 Docling/结构层保留 µ 字形或像素证据，"
            "不能靠 LLM 猜单位。",
            "",
            "### P2 成本与重复表",
            "",
            f"- 总调用 {totals['calls']}，"
            f"prompt {totals['prompt_tokens']}，"
            f"completion {totals['completion_tokens']}，"
            f"reasoning {totals['reasoning_tokens']}。",
            f"- 非 Gold scope 表提取调用占 "
            f"{totals['non_gold_table_call_ratio']:.2%}；"
            "ADI 的 table 3/11/10、AP 的 table 5/6/7、"
            "NCP 的 table 5/6/7 全部是 out-of-scope。",
            f"- retry 调用占 {totals['retry_call_ratio']:.2%}，"
            "但 `correctable_retry_corrected=0`；"
            "retry 目前只增加成本。",
            f"- length 截断 {totals['length_calls']['calls']} 次，"
            f"消耗 completion token 的 "
            f"{totals['length_completion_token_share']:.2%}；"
            "主要集中在 ADI table 3/11，不是 Gold table 4。",
            "",
            "## 3. 不建议的修复",
            "",
            "- 不要放宽 `check_min_typ_max` 让所有 invalid 变 validated；"
            "这会掩盖真实数值冲突。",
            "- 不要把 Validator 误拒计入 Recall 提升；"
            "B3.5c 的 TP 已包含这些 invalid。",
            "- 不要用 StructureRecovery 的结构覆盖率替代真实 F1；"
            "必须重新跑 LLM E2E 才能声称 Recall 改善。",
            "- 不要为 ADI table 4 直接补规则硬编码 18 条 Gold；"
            "必须先修多值拆分，再用新运行验证。",
            "",
        ]
    )

    return "\n".join(lines)


def report_markdown(trace, structure, cost, review):
    totals = cost["totals"]
    stage_counts = trace["stage_counts"]
    group_counts = trace["stage_group_counts"]
    lines = [
        "# B3.5d：50 条 FN 逐层归因与最小修复优先级",
        "",
        "## 1. 基线（只读）",
        "",
        "B3.5c Flat Baseline：TP=26、FP=2、FN=50、"
        "P=92.86%、R=34.21%、F1=50.00%、严格 F1=48.07%。",
        "Gold 76 条、预测 169 条、in-scope 预测 28 条。",
        "",
        "本报告只消费 B3.5c 运行产物、B3.5b 冻结 Gold 和 B3.5d "
        "离线结构结果；没有调用 DeepSeek，没有修改生产代码、"
        "Gold、TableSelector、StructureRecovery、Adapter、Validator。",
        "",
        "## 2. 直接回答：损失发生在哪一步",
        "",
        "| 阶段组 | FN | 占比 | 说明 |",
        "| --- | ---: | ---: | --- |",
        f"| 输入结构（Flat/Docling/StructureRecovery） | "
        f"{group_counts.get('input_structure', 0)} | "
        f"{group_counts.get('input_structure', 0) / 50:.0%} | "
        "多值未拆、µA→mA、行续接 |",
        f"| LLM 提取 | {group_counts.get('llm_extraction', 0)} | "
        f"{group_counts.get('llm_extraction', 0) / 50:.0%} | "
        "全部为省略，非截断 |",
        "| Validator | 0 | 0% | 13 条误拒命中已正确预测的 TP |",
        "| 评分匹配 | 0 | 0% | 无评分类 FN |",
        "",
        "细化：",
        "",
        "| primary_stage | FN |",
        "| --- | ---: |",
    ]

    for stage, count in sorted(
        stage_counts.items(), key=lambda item: -item[1]
    ):
        lines.append(f"| `{stage}` | {count} |")

    lines.extend(
        [
            "",
            "关键修正：B3.5c 的 44 条 "
            "`structure_recovery_row_or_unit_error` 是粗分类。"
            "逐层追踪后，这 44 条中只有 18 条是真正的结构多值未拆，"
            "22 条是 LLM 完整省略，3 条是单位源错误，1 条是行续接；"
            "再加上 B3.5c 原本标为 `llm_omitted` 的 4 条和 "
            "`llm_wrong_unit` 的 2 条 FN，最终 26 条 FN "
            "最早失败在 LLM 省略。不能把 44 条都归给结构。",
            "",
            "## 3. 分样本结论",
            "",
            "| 样本 | FN | 最早失败 | 证据 |",
            "| --- | ---: | --- | --- |",
            "| ADI LT3045 | 18 | 输入结构 | table 4 已选中，"
            "2 次调用均 stop 且 0 参数；17/18 Gold 值位于多值单元格；"
            "Recovery 0/18、Adapter 0/18 |",
            "| Diodes AP2112 | 11 | LLM 提取（10）+ 行续接（1） | "
            "10 条为 line/load regulation、shutdown、short circuit、"
            "noise 省略；1 条 PSRR f=1kHz 被 Docling 归到 "
            "Output Voltage 续行且单位为空 |",
            "| onsemi NCP170 | 11 | LLM 提取（6）+ 单位（5） | "
            "5 条 µA 被 Docling 解析为 mA/A；6 条非重点身份省略 |",
            "| Renesas ISL80510 | 10 | LLM 提取 | "
            "全部 stop 调用下省略 line/load regulation、"
            "shutdown、short circuit、noise |",
            "",
            "## 4. ADI table 4：为什么不是截断",
            "",
            "table 4 的两次调用都是 `finish_reason=stop`，"
            "返回 `{\"parameters\":[]}`，内容长度 56 字符。"
            "截断集中在 table 3（2 次 length）和 table 11"
            "（1 次 length），与 table 4 无关。"
            "因此 ADI 的 18 条 FN 不能归因于输出截断或 Selector 漏选。",
            "",
            "## 5. Validator：13 条误拒的真实影响",
            "",
            "13 条记录的内容（身份、值、单位、value_type）"
            "与 Gold 完全一致，最终却为 invalid。复核全部可复现：",
            "",
            f"- 跨电压档分组：{review['root_cause_counts'].get('cross_variant_grouping', 0)} 条；",
            f"- 表头脚注：{review['root_cause_counts'].get('ambiguous_column_header', 0)} 条；",
            f"- 多值单元格 pending：{review['root_cause_counts'].get('multi_value_cell', 0)} 条；",
            f"- 合并单元格空单位：{review['root_cause_counts'].get('merged_unit_cell', 0)} 条主因（另 2 条为次要原因）。",
            "",
            "但这些记录在 B3.5c 中**已被算作 TP**："
            "评分不按 `validation_status` 过滤。"
            "所以修复 Validator 不会提高本轮 Recall，"
            "只是修复生产端“正确预测被误标不可信”的问题。",
            "",
            "## 6. 成本",
            "",
            f"- 调用 {totals['calls']}；prompt {totals['prompt_tokens']}；"
            f"completion {totals['completion_tokens']}；"
            f"reasoning {totals['reasoning_tokens']}。",
            f"- 非 Gold scope 表提取调用占 "
            f"{totals['non_gold_table_call_ratio']:.2%}。",
            f"- retry 占 {totals['retry_call_ratio']:.2%}，"
            "invalid→validated 纠正 0。",
            f"- length 截断 {totals['length_calls']['calls']} 次，"
            f"占 completion token "
            f"{totals['length_completion_token_share']:.2%}。",
            "",
            "## 7. 修复优先级（最小集）",
            "",
            "1. **P0 StructureRecovery 多值拆分/单位对齐**："
            "解除 ADI 18 条输入阻塞，并修复 NCP PSRR、"
            "Renesas short-circuit 的 min/max 丢失。",
            "2. **P0 Prompt 身份覆盖**：覆盖 line/load regulation、"
            "shutdown、short circuit、noise、current limit 家族；"
            "上界 26 条 FN。",
            "3. **P1 Validator 正确性**：跨档分组、脚注表头、"
            "合并单位、pending 状态语义；13 条误拒，"
            "本轮 F1 影响 0。",
            "4. **P1 Docling µ 字形保真**：NCP 5 条单位 FN；"
            "不能让 LLM 猜单位。",
            "5. **P2 选择器/去重/截断/retry**：成本优化，"
            "不直接增加 Recall。",
            "",
            "## 8. 复现",
            "",
            "```bash",
            "PYTHONPATH=backend:backend/scripts .venv/bin/python "
            "backend/scripts/b35d_trace_fn.py",
            "PYTHONPATH=backend:backend/scripts .venv/bin/python "
            "backend/scripts/b35d_structure_compare.py",
            "PYTHONPATH=backend:backend/scripts .venv/bin/python "
            "backend/scripts/b35d_review_validator.py",
            "PYTHONPATH=backend:backend/scripts .venv/bin/python "
            "backend/scripts/b35d_cost.py",
            "PYTHONPATH=backend:backend/scripts .venv/bin/python "
            "backend/scripts/b35d_build_reports.py",
            "PYTHONPATH=backend .venv/bin/python -m pytest "
            "backend/tests/test_b35d_offline.py -q",
            "```",
            "",
            "## 9. 边界与限制",
            "",
            "- 单次跨型号运行，不构成统计显著结论。",
            "- 结构覆盖率为离线可恢复性，不是真实提取 F1。",
            "- 未重新调用 LLM，因此不能声称任何修复会提高 Recall；"
            "必须用新 E2E 验证。",
            "- 对 `identity_not_in_flat_prompt_focus` 只证明相关，"
            "不证明唯一因果。",
            "- 未 Git Commit。",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    trace = load_json(TRACE_DIR / "fn_attribution.json")
    structure = load_json(STRUCTURE_PATH)
    cost = load_json(COST_PATH)
    review = load_json(VALIDATOR_DIR / "review.json")

    facts = adi_facts()
    write_json(TRACE_DIR / "adi_table4.json", facts)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "adi_table4.md").write_text(
        adi_markdown(facts), encoding="utf-8"
    )
    (REPORT_DIR / "flat_vs_recovered.md").write_text(
        structure_markdown(structure, trace), encoding="utf-8"
    )
    (REPORT_DIR / "fix_priority.md").write_text(
        fix_priority_markdown(trace, structure, cost, review),
        encoding="utf-8",
    )
    (EXPERIMENT_DIR / "REPORT.md").write_text(
        report_markdown(trace, structure, cost, review),
        encoding="utf-8",
    )

    print("ADI:", TRACE_DIR / "adi_table4.json")
    print("ADI report:", REPORT_DIR / "adi_table4.md")
    print(
        "Structure report:",
        REPORT_DIR / "flat_vs_recovered.md",
    )
    print(
        "Fix priority:", REPORT_DIR / "fix_priority.md"
    )
    print("Main report:", EXPERIMENT_DIR / "REPORT.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
