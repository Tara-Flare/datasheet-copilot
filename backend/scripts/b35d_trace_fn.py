"""B3.5d：对 B3.5c 的 50 条 FN 做逐层证据链归因（只读，不调用 LLM）。

阶段顺序：
TableSelector -> TableParser/Flat 输入 -> LLM 原始响应 -> 最终预测
-> Validator -> 评分匹配。

只有能从已保存产物直接证实的原因才写入 primary_stage；
无法区分时写 unknown，不做推测。
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from b35c_score_flat import (  # noqa: E402
    content_allowed,
    normalize_gold,
    normalize_prediction,
    normalize_unit,
)
from b35d_common import (  # noqa: E402
    B35C_ERRORS,
    EXPERIMENT_DIR,
    FLAT_PROMPT_FOCUS,
    TRACE_DIR,
    attach_events,
    identity_matches_text,
    load_gold,
    load_json,
    load_logical,
    load_manifest,
    normalize_number,
    number_tokens,
    recovered_row_matches_identity,
    row_cells_for_gold,
    table_rows,
    unit_column_index,
    unit_contains,
    value_column_index,
    value_contains,
    value_is_single,
    write_json,
)


STAGE_GROUP = {
    "table_selector_miss": "table_selector",
    "flat_input_value_missing": "input_structure",
    "structure_recovery_gap": "input_structure",
    "table_parser_unit_error": "input_structure",
    "table_parser_row_continuation_error": "input_structure",
    "llm_omitted": "llm_extraction",
    "llm_output_truncated": "llm_extraction",
    "llm_wrong_value": "llm_extraction",
    "llm_wrong_unit": "llm_extraction",
    "llm_wrong_value_type": "llm_extraction",
    "llm_wrong_identity": "llm_extraction",
    "retry_or_merge_lost": "validator_retry",
    "validator_dropped": "validator_retry",
    "scoring_mismatch": "scoring",
    "unknown": "unknown",
}


def _input_identity_text(table):
    columns = [
        str(column).strip().lower()
        for column in table.get("columns", [])
    ]
    indices = [
        index
        for index, column in enumerate(columns)
        if "parameter" in column or "symbol" in column
    ]

    return indices


def flat_input_snapshot(sample_id, gold):
    table = table_rows(sample_id, gold["table_id"])

    if not table:
        return {"table_present": False}

    row = row_cells_for_gold(
        sample_id, gold["table_id"], gold["row_index"]
    )

    if row is None:
        return {"table_present": True, "row_present": False}

    value_index = value_column_index(table, gold["value_type"])
    unit_index = unit_column_index(table)
    value_cell = (
        row[value_index]
        if value_index is not None and value_index < len(row)
        else None
    )
    unit_cell = (
        row[unit_index]
        if unit_index is not None and unit_index < len(row)
        else None
    )
    identity_indices = _input_identity_text(table)
    identity_text = " ".join(
        str(row[index])
        for index in identity_indices
        if index < len(row)
    )
    conditions_text = " ".join(
        str(value)
        for index, value in enumerate(row)
        if index not in {value_index, unit_index}
        and value is not None
    )
    all_units = " ".join(str(cell or "") for cell in row)
    value_numbers = number_tokens(value_cell)
    type_cells = []

    for value_type in ("min", "typ", "max"):
        type_index = value_column_index(table, value_type)

        if type_index is not None and type_index < len(row):
            type_cells.append(row[type_index])

    row_multi_value = any(
        len(number_tokens(cell)) > 1 for cell in type_cells
    )
    unit_values = [
        str(cell).strip()
        for index, cell in enumerate(row)
        if index == unit_index and str(cell or "").strip()
    ]

    return {
        "table_present": True,
        "row_present": True,
        "columns": table.get("columns", []),
        "row": row,
        "value_column_index": value_index,
        "value_cell": value_cell,
        "unit_column_index": unit_index,
        "unit_cell": unit_cell,
        "identity_text": identity_text,
        "conditions_text": conditions_text,
        "row_identity_match": (
            identity_matches_text(
                gold["identity"], identity_text
            )
            or identity_matches_text(
                gold["identity"],
                " ".join(str(cell or "") for cell in row),
            )
        ),
        "value_in_input": value_contains(value_cell, gold["value"]),
        "value_is_single": value_is_single(value_cell),
        "value_number_count": len(value_numbers),
        "unit_in_input": unit_contains(unit_cell, gold["unit"]),
        "unit_cell_blank": not str(unit_cell or "").strip(),
        "unit_source_mismatch": bool(
            str(unit_cell or "").strip()
        )
        and not unit_contains(unit_cell, gold["unit"]),
        "unit_values": unit_values,
        "row_unit_token_count": len(
            [
                token
                for token in all_units.split()
                if token and token not in {"-", "—"}
            ]
        ),
        "multi_value_cell": len(value_numbers) > 1,
        "row_multi_value": row_multi_value,
        "multi_unit_row": len(unit_values) > 1,
    }


def recovered_snapshot(sample_id, gold):
    logical = load_logical(sample_id, gold["table_id"])

    if not logical:
        return {"available": False}

    candidates = [
        row
        for row in logical.get("logical_rows", [])
        if gold["row_index"]
        in (row.get("source_row_indices") or [])
    ]

    if not candidates:
        return {
            "available": True,
            "row_found": False,
            "logical_file": str(
                "data/outputs/experiments/B3.5d/structure/logical"
            ),
        }

    rows = []

    for row in candidates:
        identity_ok, similarity = recovered_row_matches_identity(
            gold, row
        )
        field_name = gold["value_type"]
        field_value = row.get(field_name, "")
        unit_value = row.get("unit", "")
        value_ok = value_contains(field_value, gold["value"])
        single = value_is_single(field_value)
        unit_ok = unit_contains(unit_value, gold["unit"])
        rows.append(
            {
                "logical_row_id": row.get("logical_row_id"),
                "source_row_indices": row.get(
                    "source_row_indices", []
                ),
                "parameter_symbol": row.get(
                    "parameter_symbol", ""
                ),
                "parameter_name": row.get(
                    "parameter_name", ""
                ),
                "test_conditions": row.get(
                    "test_conditions", ""
                ),
                "min": row.get("min", ""),
                "typ": row.get("typ", ""),
                "max": row.get("max", ""),
                "unit": unit_value,
                "identity_match": identity_ok,
                "symbol_similarity": round(similarity, 3),
                "value_field": field_name,
                "value_field_value": field_value,
                "value_present": value_ok,
                "value_single": single,
                "value_number_count": len(
                    number_tokens(field_value)
                ),
                "unit_present": unit_ok,
                "needs_review": bool(row.get("needs_review")),
                "complete": bool(
                    identity_ok and value_ok and single and unit_ok
                ),
            }
        )

    return {
        "available": True,
        "row_found": True,
        "rows": rows,
        "any_complete": any(row["complete"] for row in rows),
        "any_identity": any(
            row["identity_match"] for row in rows
        ),
        "any_value": any(row["value_present"] for row in rows),
        "any_unit": any(row["unit_present"] for row in rows),
        "any_multi_value": any(
            row["value_number_count"] > 1 for row in rows
        ),
    }


def _raw_param_matches(param, gold, prediction):
    return (
        prediction["table_id"] == gold["table_id"]
        and prediction["row_index"] == gold["row_index"]
    )


def _summarize_raw_params(params, gold, predictions):
    same_row = []
    same_identity = []
    exact = []

    for index, param in enumerate(params):
        prediction = predictions[index]

        if _raw_param_matches(param, gold, prediction):
            same_row.append(index)

            if prediction["identity"] == gold["identity"]:
                same_identity.append(index)

                if (
                    prediction["value_type"] == gold["value_type"]
                    and value_contains(
                        prediction["value"], gold["value"]
                    )
                    and normalize_unit(prediction["unit"])
                    == gold["unit"]
                ):
                    exact.append(index)

    return {
        "same_row_indices": same_row,
        "same_identity_indices": same_identity,
        "exact_indices": exact,
        "params": [
            {
                "index": index,
                "name": param.get("name"),
                "value": param.get("value"),
                "unit": param.get("unit"),
                "value_type": param.get("value_type"),
                "test_conditions": param.get(
                    "test_conditions", ""
                ),
                "table_id": (param.get("evidence") or {}).get(
                    "table_id"
                ),
                "row_index": (
                    param.get("evidence") or {}
                ).get("row_index"),
                "source_text": (
                    param.get("evidence") or {}
                ).get("source_text", ""),
            }
            for index in sorted(
                set(same_row) | set(same_identity) | set(exact)
            )
            for param in [params[index]]
        ],
    }


def raw_snapshot(sample_id, gold, events):
    covering = [
        event
        for event in events
        if event["kind"] == "extract"
        and event["table_id"] == gold["table_id"]
        and event["start"] <= gold["row_index"]
        < event["start"] + event["rows"]
    ]
    retry_events = [
        event
        for event in events
        if event["kind"] == "retry"
    ]
    all_params = []
    contexts = []

    for event in covering:
        params = event.get("parameters", [])
        predictions = [
            normalize_prediction(param, index)
            for index, param in enumerate(params)
        ]
        all_params.extend(params)
        summary = _summarize_raw_params(
            params, gold, predictions
        )
        contexts.append(
            {
                "index": event["index"],
                "start": event["start"],
                "rows": event["rows"],
                "depth": event["depth"],
                "finish": event["finish"],
                "content_length": event["content_length"],
                "json_valid": event.get("json_valid"),
                "parameter_count": len(params),
                "same_row_indices": summary["same_row_indices"],
                "same_identity_indices": summary[
                    "same_identity_indices"
                ],
                "exact_indices": summary["exact_indices"],
                "partial_identity_mention": (
                    not event.get("json_valid")
                    and bool(event.get("content"))
                    and any(
                        identity_matches_text(
                            gold["identity"],
                            event.get("content", ""),
                        )
                        for _ in [0]
                    )
                ),
            }
        )

    retry_predictions = []
    retry_params = []

    for event in retry_events:
        for param in event.get("parameters", []):
            retry_params.append(param)

    retry_normalized = [
        normalize_prediction(param, index)
        for index, param in enumerate(retry_params)
    ]
    retry_summary = _summarize_raw_params(
        retry_params, gold, retry_normalized
    )

    predictions = [
        normalize_prediction(param, index)
        for index, param in enumerate(all_params)
    ]
    combined = _summarize_raw_params(
        all_params, gold, predictions
    )

    return {
        "covering_calls": contexts,
        "truncated_calls": sum(
            1 for event in covering if event["finish"] == "length"
        ),
        "stop_calls": sum(
            1 for event in covering if event["finish"] == "stop"
        ),
        "empty_content_calls": sum(
            1
            for event in covering
            if event["content_length"] == 0
        ),
        "parameters_covering": len(all_params),
        "same_row_indices": combined["same_row_indices"],
        "same_identity_indices": combined[
            "same_identity_indices"
        ],
        "exact_indices": combined["exact_indices"],
        "params": combined["params"],
        "retry_call_count": len(retry_events),
        "retry_parameter_count": len(retry_params),
        "retry_same_row_indices": retry_summary[
            "same_row_indices"
        ],
        "retry_same_identity_indices": retry_summary[
            "same_identity_indices"
        ],
        "retry_exact_indices": retry_summary["exact_indices"],
        "retry_params": retry_summary["params"],
        "identity_any": bool(combined["same_identity_indices"]),
        "same_row_any": bool(combined["same_row_indices"]),
        "exact_any": bool(combined["exact_indices"]),
    }


def final_snapshot(sample_id, gold):
    validation = load_json(
        EXPERIMENT_ROOT
        / "B3.5c/runs/flat_baseline_r1"
        / sample_id
        / "validation.json"
    )
    params = validation.get("parameters", [])
    predictions = [
        normalize_prediction(param, index)
        for index, param in enumerate(params)
    ]
    same_row = []
    same_identity = []
    exact = []

    for index, param in enumerate(params):
        prediction = predictions[index]

        if (
            prediction["table_id"] == gold["table_id"]
            and prediction["row_index"] == gold["row_index"]
        ):
            same_row.append(index)

            if prediction["identity"] == gold["identity"]:
                same_identity.append(index)

                if (
                    prediction["value_type"] == gold["value_type"]
                    and value_contains(
                        prediction["value"], gold["value"]
                    )
                    and normalize_unit(prediction["unit"])
                    == gold["unit"]
                ):
                    exact.append(index)

    return {
        "total_parameters": len(params),
        "same_row_indices": same_row,
        "same_identity_indices": same_identity,
        "exact_indices": exact,
        "exact_any": bool(exact),
        "same_identity_params": [
            {
                "index": index,
                "name": params[index].get("name"),
                "value": params[index].get("value"),
                "unit": params[index].get("unit"),
                "value_type": params[index].get("value_type"),
                "test_conditions": params[index].get(
                    "test_conditions", ""
                ),
                "table_id": (
                    params[index].get("evidence") or {}
                ).get("table_id"),
                "row_index": (
                    params[index].get("evidence") or {}
                ).get("row_index"),
                "validation_status": params[index].get(
                    "validation_status"
                ),
                "validation_errors": params[index].get(
                    "validation_errors", []
                ),
            }
            for index in sorted(set(same_identity) | set(exact))
        ],
    }


EXPERIMENT_ROOT = PROJECT_ROOT / "data/outputs/experiments"


def classify_primary(
    gold,
    flat,
    recovered,
    raw,
    final,
    selected_table_ids,
):
    if gold["table_id"] not in selected_table_ids:
        return "table_selector_miss", ["scope table 未选中"]

    if not flat.get("row_present"):
        return "flat_input_value_missing", ["Gold 来源行不在 Flat 输入"]

    if not flat.get("value_in_input"):
        return "flat_input_value_missing", [
            "Flat 输入来源列不包含 Gold 数值"
        ]

    # 行续接错误：输入行的参数身份不是 Gold 身份，且 Symbol 为空。
    if not flat.get("row_identity_match"):
        row = flat.get("row") or []
        symbol = str(row[0] if row else "").strip()
        return "table_parser_row_continuation_error", [
            "Flat 输入行被 Docling 归到其他参数名下",
            f"symbol={symbol!r}",
            f"identity_text={flat.get('identity_text')!r}",
        ]

    if flat.get("unit_source_mismatch"):
        return "table_parser_unit_error", [
            f"输入单位 {flat.get('unit_cell')!r} 与 Gold "
            f"单位 {gold['unit']!r} 不一致"
        ]

    if (
        flat.get("row_multi_value")
        and not recovered.get("any_complete")
    ):
        return "structure_recovery_gap", [
            "来源单元格包含多个数值或重复单位",
            "离线 StructureRecovery 未恢复出单一可验证值",
            f"recovered_complete={recovered.get('any_complete')}",
        ]

    if final.get("exact_any"):
        return "scoring_mismatch", [
            "最终预测含完全一致记录，但评分未匹配"
        ]

    if raw.get("exact_any") and not final.get("exact_any"):
        return "retry_or_merge_lost", [
            "LLM 原始响应含完全一致记录，最终预测中丢失"
        ]

    if raw.get("same_identity_indices"):
        raw_params = raw.get("params", [])

        for item in raw_params:
            if normalize_unit(item.get("unit")) != gold["unit"]:
                return "llm_wrong_unit", [
                    "LLM 同身份记录单位错误",
                    f"raw_unit={item.get('unit')!r}",
                ]

            if item.get("value_type") != gold["value_type"]:
                return "llm_wrong_value_type", [
                    "LLM 同身份记录 value_type 错误"
                ]

        return "llm_wrong_value", [
            "LLM 同身份记录数值错误"
        ]

    if raw.get("truncated_calls") and not raw.get("stop_calls"):
        return "llm_output_truncated", [
            "覆盖该行的所有调用均因 finish_reason=length 截断"
        ]

    return "llm_omitted", [
        "覆盖该行的调用未截断，但原始响应没有该身份参数"
    ]


def build_trace():
    gold = load_gold()
    manifest = load_manifest()
    manifest_samples = {
        sample["sample_id"]: sample
        for sample in manifest["samples"]
    }
    source_errors = load_json(B35C_ERRORS)
    fn_errors = [
        error for error in source_errors if error["kind"] == "FN"
    ]
    gold_index = {}

    for sample in gold["samples"]:
        for record in sample["records"]:
            gold_index[record["gold_id"]] = normalize_gold(record)

    by_sample = defaultdict(list)

    for error in fn_errors:
        by_sample[error["sample_id"]].append(error)

    records = []

    for sample in gold["samples"]:
        sample_id = sample["sample_id"]
        events = attach_events(sample_id)
        selected = load_json(
            EXPERIMENT_ROOT
            / "B3.5c/runs/flat_baseline_r1"
            / sample_id
            / "selected_tables.json"
        )
        selected_ids = {
            table["table_id"] for table in selected["tables"]
        }
        sample_errors = {
            error["gold_id"]: error
            for error in by_sample.get(sample_id, [])
        }

        for record in sample["records"]:
            gold_record = normalize_gold(record)

            if gold_record["gold_id"] not in sample_errors:
                continue

            error = sample_errors[gold_record["gold_id"]]
            flat = flat_input_snapshot(
                sample_id, gold_record
            )
            recovered = recovered_snapshot(
                sample_id, gold_record
            )
            raw = raw_snapshot(
                sample_id, gold_record, events
            )
            final = final_snapshot(
                sample_id, gold_record
            )
            primary, reasons = classify_primary(
                gold_record,
                flat,
                recovered,
                raw,
                final,
                selected_ids,
            )
            prompt_scope = (
                gold_record["identity"] not in FLAT_PROMPT_FOCUS
            )
            secondary = []

            if prompt_scope:
                secondary.append("identity_not_in_flat_prompt_focus")

            if flat.get("row_multi_value"):
                secondary.append("flat_input_multi_value_cell")

            if flat.get("unit_source_mismatch"):
                secondary.append("docling_unit_mismatch")

            if flat.get("unit_cell_blank"):
                secondary.append("docling_unit_cell_blank")

            if not flat.get("row_identity_match"):
                secondary.append("docling_row_continuation")

            if recovered.get("available") and (
                not recovered.get("any_complete")
            ):
                secondary.append("structure_recovery_incomplete")

            if raw.get("truncated_calls"):
                secondary.append("raw_output_truncated_in_covering_chunk")

            if raw.get("identity_any"):
                secondary.append("llm_raw_identity_present")

            if raw.get("exact_any"):
                secondary.append("llm_raw_exact_present")

            if final.get("exact_any"):
                secondary.append("final_exact_present")

            if error.get("category") != primary:
                secondary.append(
                    f"b35c_category={error.get('category')}"
                )

            records.append(
                {
                    "sample_id": sample_id,
                    "vendor": sample["vendor"],
                    "model": sample["model"],
                    "gold_id": gold_record["gold_id"],
                    "gold_table_id": gold_record["table_id"],
                    "gold_row_index": gold_record["row_index"],
                    "gold_page": gold_record["page"],
                    "gold_identity": gold_record["identity"],
                    "gold_value": gold_record["value"],
                    "gold_unit": gold_record["unit"],
                    "gold_value_type": gold_record["value_type"],
                    "gold_conditions": gold_record[
                        "test_conditions"
                    ],
                    "gold_parameter_symbol": gold_record[
                        "parameter_symbol"
                    ],
                    "gold_row_text": gold_record["row_text"],
                    "b35c_category": error.get("category"),
                    "primary_stage": primary,
                    "stage_group": STAGE_GROUP[primary],
                    "primary_reasons": reasons,
                    "secondary_flags": secondary,
                    "identity_in_flat_prompt_focus": (
                        not prompt_scope
                    ),
                    "table_selected": (
                        gold_record["table_id"] in selected_ids
                    ),
                    "flat_input": flat,
                    "structure_recovery": recovered,
                    "raw_llm": raw,
                    "final_predictions": final,
                }
            )

    # 校验每条 FN 的 primary_stage 都在已知集合内。
    unknown = [
        record
        for record in records
        if record["primary_stage"] not in STAGE_GROUP
    ]

    if unknown:
        raise SystemExit(
            f"存在未知 primary_stage: {unknown[:3]}"
        )

    stage_counts = Counter(
        record["primary_stage"] for record in records
    )
    group_counts = Counter(
        record["stage_group"] for record in records
    )
    per_sample = defaultdict(Counter)
    per_sample_group = defaultdict(Counter)

    for record in records:
        per_sample[record["sample_id"]][
            record["primary_stage"]
        ] += 1
        per_sample_group[record["sample_id"]][
            record["stage_group"]
        ] += 1

    summary = {
        "schema": "b35d_fn_attribution/v1",
        "source": "B3.5c flat_baseline_r1",
        "gold": "B3.5b gold.frozen.json (76 confirmed)",
        "fn_total": len(records),
        "stage_counts": dict(stage_counts),
        "stage_group_counts": dict(group_counts),
        "per_sample_stage_counts": {
            key: dict(value)
            for key, value in per_sample.items()
        },
        "per_sample_group_counts": {
            key: dict(value)
            for key, value in per_sample_group.items()
        },
        "prompt_focus_counts": {
            "identity_in_prompt_focus": sum(
                1
                for record in records
                if record["identity_in_flat_prompt_focus"]
            ),
            "identity_not_in_prompt_focus": sum(
                1
                for record in records
                if not record["identity_in_flat_prompt_focus"]
            ),
        },
        "notes": [
            "primary_stage 只记录最早可证实的失败阶段。",
            "structure_recovery_gap 表示 Flat 输入单元格含多值，"
            "且离线 StructureRecovery 未恢复出单一可验证值；"
            "不代表 Baseline 实际调用了 StructureRecovery。",
            "table_parser_unit_error 表示 Docling 输入单位与冻结 "
            "Gold 不一致；NCP170 的 µA→mA 有 Gold 逐字形备注证据。",
            "scoring_mismatch 为 0 表示没有 FN 由评分匹配造成。",
        ],
        "records": records,
    }

    return summary


def write_csv(records, path):
    fieldnames = [
        "sample_id",
        "vendor",
        "gold_id",
        "gold_table_id",
        "gold_row_index",
        "gold_page",
        "gold_identity",
        "gold_value",
        "gold_unit",
        "gold_value_type",
        "gold_conditions",
        "b35c_category",
        "primary_stage",
        "stage_group",
        "identity_in_flat_prompt_focus",
        "table_selected",
        "flat_value_cell",
        "flat_unit_cell",
        "flat_multi_value_cell",
        "recovered_any_complete",
        "raw_covering_calls",
        "raw_same_identity",
        "raw_exact",
        "final_same_identity",
        "final_exact",
        "primary_reasons",
        "secondary_flags",
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
                    "vendor": record["vendor"],
                    "gold_id": record["gold_id"],
                    "gold_table_id": record["gold_table_id"],
                    "gold_row_index": record["gold_row_index"],
                    "gold_page": record["gold_page"],
                    "gold_identity": record["gold_identity"],
                    "gold_value": record["gold_value"],
                    "gold_unit": record["gold_unit"],
                    "gold_value_type": record[
                        "gold_value_type"
                    ],
                    "gold_conditions": record["gold_conditions"],
                    "b35c_category": record["b35c_category"],
                    "primary_stage": record["primary_stage"],
                    "stage_group": record["stage_group"],
                    "identity_in_flat_prompt_focus": record[
                        "identity_in_flat_prompt_focus"
                    ],
                    "table_selected": record["table_selected"],
                    "flat_value_cell": record["flat_input"].get(
                        "value_cell"
                    ),
                    "flat_unit_cell": record["flat_input"].get(
                        "unit_cell"
                    ),
                    "flat_multi_value_cell": record[
                        "flat_input"
                    ].get("row_multi_value"),
                    "recovered_any_complete": record[
                        "structure_recovery"
                    ].get("any_complete"),
                    "raw_covering_calls": len(
                        record["raw_llm"]["covering_calls"]
                    ),
                    "raw_same_identity": bool(
                        record["raw_llm"][
                            "same_identity_indices"
                        ]
                    ),
                    "raw_exact": bool(
                        record["raw_llm"]["exact_indices"]
                    ),
                    "final_same_identity": bool(
                        record["final_predictions"][
                            "same_identity_indices"
                        ]
                    ),
                    "final_exact": record["final_predictions"][
                        "exact_any"
                    ],
                    "primary_reasons": " | ".join(
                        record["primary_reasons"]
                    ),
                    "secondary_flags": " | ".join(
                        record["secondary_flags"]
                    ),
                }
            )


def main():
    summary = build_trace()
    output = TRACE_DIR / "fn_attribution.json"
    write_json(output, summary)
    write_csv(
        summary["records"],
        TRACE_DIR / "fn_attribution.csv",
    )

    print("FN 归因:", output)
    print("FN 总数:", summary["fn_total"])
    print("阶段:", summary["stage_counts"])
    print("阶段组:", summary["stage_group_counts"])
    print("prompt focus:", summary["prompt_focus_counts"])

    for sample_id, counts in summary[
        "per_sample_stage_counts"
    ].items():
        print(" ", sample_id, counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
