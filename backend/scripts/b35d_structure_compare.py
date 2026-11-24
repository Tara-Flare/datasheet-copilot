"""B3.5d：Flat 输入 vs StructureRecovery vs LogicalTableAdapter 离线对照。

不调用 LLM。所有结构指标只描述“离线可恢复性/可覆盖率”，
不得当作真实提取 F1 改善。
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.extraction.logical_table_adapter import (  # noqa: E402
    adapt_logical_rows,
)
from b35c_score_flat import (  # noqa: E402
    conditions_complete,
    normalize_gold,
    normalize_unit,
)
from b35d_common import (  # noqa: E402
    EXPERIMENT_DIR,
    LOGICAL_DIR,
    load_gold,
    load_json,
    load_logical,
    load_raw_structure,
    normalize_number,
    number_tokens,
    recovered_row_matches_identity,
    table_rows,
    unit_contains,
    value_column_index,
    value_contains,
    value_is_single,
    write_json,
)


TARGETS = {
    "adi_lt3045": 4,
    "diodes_ap2112": 4,
    "onsemi_ncp170": 4,
    "renesas_isl80510": 7,
}

NAME_BY_SAMPLE = {
    "adi_lt3045": "adi_lt3045.table4",
    "diodes_ap2112": "diodes_ap2112.table4",
    "onsemi_ncp170": "onsemi_ncp170.table4",
    "renesas_isl80510": "renesas_isl80510.table7",
}


def build_adapter(sample_id):
    logical = load_logical(sample_id, TARGETS[sample_id])
    raw = load_raw_structure(sample_id)

    if logical is None or raw is None:
        return None

    adapted = adapt_logical_rows(
        logical,
        raw_structure=raw,
        table_ids=[TARGETS[sample_id]],
    )
    write_json(
        LOGICAL_DIR
        / f"{NAME_BY_SAMPLE[sample_id]}.adapter.json",
        adapted,
    )

    return adapted


def recovered_rows_for(record, logical):
    return [
        row
        for row in logical.get("logical_rows", [])
        if record["row_index"]
        in (row.get("source_row_indices") or [])
    ]


def adapter_rows_for(record, adapted):
    if not adapted:
        return []

    table = adapted["tables"][0]

    return [
        row
        for row in table.get("records", [])
        if record["row_index"] in (row.get("source_row_indices") or [])
    ]


def row_to_prediction(row, gold):
    return {
        "identity": gold["identity"],
        "value_type": gold["value_type"],
        "value": gold["value"],
        "unit": gold["unit"],
        "test_conditions": row.get("test_conditions", ""),
    }


def analyze_record(record, logical, adapted):
    gold = normalize_gold(record)
    recovered = recovered_rows_for(gold, logical)
    adapter = adapter_rows_for(gold, adapted)
    recovered_details = []

    for row in recovered:
        identity_ok, similarity = recovered_row_matches_identity(
            gold, row
        )
        field_name = gold["value_type"]
        field_value = row.get(field_name, "")
        unit_value = row.get("unit", "")
        value_ok = value_contains(field_value, gold["value"])
        single = value_is_single(field_value)
        unit_ok = unit_contains(unit_value, gold["unit"])
        recovered_details.append(
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
                "needs_review": bool(
                    row.get("needs_review")
                ),
                "complete": bool(
                    identity_ok and value_ok and single and unit_ok
                ),
            }
        )

    adapter_details = []

    for row in adapter:
        identity_ok, similarity = recovered_row_matches_identity(
            gold, row
        )
        field_name = gold["value_type"]
        field_value = row.get(field_name, "")
        value_ok = value_contains(field_value, gold["value"])
        single = value_is_single(field_value)
        unit_ok = unit_contains(row.get("unit", ""), gold["unit"])
        condition_ok = conditions_complete(
            row_to_prediction(row, gold), gold
        )
        adapter_details.append(
            {
                "logical_row_id": row.get("logical_row_id"),
                "source_row_indices": row.get(
                    "source_row_indices", []
                ),
                "primary_row_index": row.get(
                    "primary_row_index"
                ),
                "parameter_symbol": row.get(
                    "parameter_symbol", ""
                ),
                "test_conditions": row.get(
                    "test_conditions", ""
                ),
                "min": row.get("min", ""),
                "typ": row.get("typ", ""),
                "max": row.get("max", ""),
                "unit": row.get("unit", ""),
                "identity_match": identity_ok,
                "symbol_similarity": round(similarity, 3),
                "value_field": field_name,
                "value_field_value": field_value,
                "value_present": value_ok,
                "value_single": single,
                "unit_present": unit_ok,
                "condition_complete": condition_ok,
                "row_index_correct": (
                    row.get("primary_row_index")
                    == gold["row_index"]
                    or gold["row_index"]
                    in (row.get("source_row_indices") or [])
                ),
                "complete": bool(
                    identity_ok
                    and value_ok
                    and single
                    and unit_ok
                    and condition_ok
                ),
            }
        )

    return {
        "gold_id": gold["gold_id"],
        "sample_id": gold["gold_id"].split(":")[0],
        "table_id": gold["table_id"],
        "row_index": gold["row_index"],
        "page": gold["page"],
        "identity": gold["identity"],
        "value": gold["value"],
        "unit": gold["unit"],
        "value_type": gold["value_type"],
        "conditions": gold["test_conditions"],
        "row_text": gold["row_text"],
        "is_gold_table": True,
        "structure_recovery": {
            "row_found": bool(recovered),
            "rows": recovered_details,
            "any_complete": any(
                row["complete"] for row in recovered_details
            ),
            "any_identity": any(
                row["identity_match"]
                for row in recovered_details
            ),
            "any_value": any(
                row["value_present"]
                for row in recovered_details
            ),
            "any_unit": any(
                row["unit_present"] for row in recovered_details
            ),
            "any_multi_value": any(
                row["value_number_count"] > 1
                for row in recovered_details
            ),
        },
        "adapter": {
            "row_found": bool(adapter),
            "rows": adapter_details,
            "any_complete": any(
                row["complete"] for row in adapter_details
            ),
            "any_identity": any(
                row["identity_match"]
                for row in adapter_details
            ),
            "any_value": any(
                row["value_present"]
                for row in adapter_details
            ),
            "any_unit": any(
                row["unit_present"] for row in adapter_details
            ),
            "any_condition_complete": any(
                row["condition_complete"]
                for row in adapter_details
            ),
            "any_row_index_correct": any(
                row["row_index_correct"]
                for row in adapter_details
            ),
        },
    }


def flat_rows_for_gold_table(sample_id, gold_records):
    table = table_rows(sample_id, TARGETS[sample_id])
    rows = table.get("rows", [])
    multi_value_rows = []

    for index, row in enumerate(rows):
        cells = []

        for value_type in ("min", "typ", "max"):
            column = value_column_index(table, value_type)

            if column is not None and column < len(row):
                cells.append(row[column])

        if any(len(number_tokens(cell)) > 1 for cell in cells):
            multi_value_rows.append(index)

    return {
        "table_id": TARGETS[sample_id],
        "columns": table.get("columns", []),
        "row_count": len(rows),
        "multi_value_rows": multi_value_rows,
        "multi_value_row_count": len(multi_value_rows),
        "gold_row_count": len(gold_records),
        "gold_rows_multi_value": sum(
            1
            for record in gold_records
            if record["row_index"] in multi_value_rows
        ),
    }


def summarize(sample, records):
    total = len(records)
    recovered_complete = sum(
        1 for record in records
        if record["structure_recovery"]["any_complete"]
    )
    adapter_complete = sum(
        1 for record in records
        if record["adapter"]["any_complete"]
    )
    adapter_found = sum(
        1 for record in records if record["adapter"]["row_found"]
    )
    adapter_identity = sum(
        1
        for record in records
        if record["adapter"]["any_identity"]
    )
    adapter_value = sum(
        1
        for record in records
        if record["adapter"]["any_value"]
    )
    adapter_unit = sum(
        1
        for record in records
        if record["adapter"]["any_unit"]
    )
    adapter_conditions = sum(
        1
        for record in records
        if record["adapter"]["any_condition_complete"]
    )
    adapter_row_index = sum(
        1
        for record in records
        if record["adapter"]["any_row_index_correct"]
    )

    return {
        "sample_id": sample["sample_id"],
        "vendor": sample["vendor"],
        "model": sample["model"],
        "gold_records": total,
        "recovered_rows_complete": recovered_complete,
        "recovered_coverage": round(
            recovered_complete / total, 4
        )
        if total
        else None,
        "adapter_rows_matched": adapter_found,
        "adapter_identity_matched": adapter_identity,
        "adapter_value_present": adapter_value,
        "adapter_unit_present": adapter_unit,
        "adapter_conditions_complete": adapter_conditions,
        "adapter_row_index_correct": adapter_row_index,
        "adapter_complete": adapter_complete,
        "adapter_coverage": round(adapter_complete / total, 4)
        if total
        else None,
        "adapter_identity_coverage": round(
            adapter_identity / total, 4
        )
        if total
        else None,
        "adapter_value_coverage": round(
            adapter_value / total, 4
        )
        if total
        else None,
        "adapter_unit_rate": round(
            adapter_unit / adapter_found, 4
        )
        if adapter_found
        else None,
        "adapter_condition_rate": round(
            adapter_conditions / adapter_found, 4
        )
        if adapter_found
        else None,
        "adapter_row_index_rate": round(
            adapter_row_index / adapter_found, 4
        )
        if adapter_found
        else None,
    }


def write_csv(records, path):
    fieldnames = [
        "sample_id",
        "gold_id",
        "table_id",
        "row_index",
        "page",
        "identity",
        "value",
        "unit",
        "value_type",
        "conditions",
        "recovered_row_found",
        "recovered_complete",
        "recovered_multi_value",
        "adapter_row_found",
        "adapter_identity",
        "adapter_value",
        "adapter_unit",
        "adapter_condition_complete",
        "adapter_row_index_correct",
        "adapter_complete",
        "recovered_logical_row_ids",
        "adapter_logical_row_ids",
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
                    "table_id": record["table_id"],
                    "row_index": record["row_index"],
                    "page": record["page"],
                    "identity": record["identity"],
                    "value": record["value"],
                    "unit": record["unit"],
                    "value_type": record["value_type"],
                    "conditions": record["conditions"],
                    "recovered_row_found": record[
                        "structure_recovery"
                    ]["row_found"],
                    "recovered_complete": record[
                        "structure_recovery"
                    ]["any_complete"],
                    "recovered_multi_value": record[
                        "structure_recovery"
                    ]["any_multi_value"],
                    "adapter_row_found": record["adapter"][
                        "row_found"
                    ],
                    "adapter_identity": record["adapter"][
                        "any_identity"
                    ],
                    "adapter_value": record["adapter"][
                        "any_value"
                    ],
                    "adapter_unit": record["adapter"][
                        "any_unit"
                    ],
                    "adapter_condition_complete": record[
                        "adapter"
                    ]["any_condition_complete"],
                    "adapter_row_index_correct": record[
                        "adapter"
                    ]["any_row_index_correct"],
                    "adapter_complete": record["adapter"][
                        "any_complete"
                    ],
                    "recovered_logical_row_ids": ",".join(
                        str(row["logical_row_id"])
                        for row in record[
                            "structure_recovery"
                        ]["rows"]
                    ),
                    "adapter_logical_row_ids": ",".join(
                        str(row["logical_row_id"])
                        for row in record["adapter"]["rows"]
                    ),
                }
            )


def main():
    gold = load_gold()
    all_records = []
    samples = []

    for sample in gold["samples"]:
        sample_id = sample["sample_id"]
        logical = load_logical(sample_id, TARGETS[sample_id])

        if logical is None:
            raise SystemExit(
                f"缺少 {sample_id} 的离线 logical_rows"
            )

        adapted = build_adapter(sample_id)
        records = [
            analyze_record(record, logical, adapted)
            for record in sample["records"]
        ]
        all_records.extend(records)
        summary = summarize(sample, records)
        table = adapted["tables"][0]
        summary.update(
            {
                "table_id": TARGETS[sample_id],
                "logical_rows_total": len(
                    logical.get("logical_rows", [])
                ),
                "logical_needs_review": sum(
                    1
                    for row in logical.get("logical_rows", [])
                    if row.get("needs_review")
                ),
                "adapter_rows_total": table.get("row_count"),
                "adapter_excluded_rows": len(
                    table.get("excluded_rows", [])
                ),
                "adapter_excluded_reasons": dict(
                    Counter(
                        row.get("reason")
                        for row in table.get(
                            "excluded_rows", []
                        )
                    )
                ),
                "flat_table": flat_rows_for_gold_table(
                    sample_id,
                    [
                        normalize_gold(record)
                        for record in sample["records"]
                    ],
                ),
            }
        )
        samples.append(summary)

    totals = {
        "gold_records": sum(
            sample["gold_records"] for sample in samples
        ),
        "recovered_rows_complete": sum(
            sample["recovered_rows_complete"]
            for sample in samples
        ),
        "adapter_rows_matched": sum(
            sample["adapter_rows_matched"]
            for sample in samples
        ),
        "adapter_complete": sum(
            sample["adapter_complete"] for sample in samples
        ),
        "adapter_identity_matched": sum(
            sample["adapter_identity_matched"]
            for sample in samples
        ),
        "adapter_value_present": sum(
            sample["adapter_value_present"]
            for sample in samples
        ),
        "adapter_unit_present": sum(
            sample["adapter_unit_present"]
            for sample in samples
        ),
        "adapter_conditions_complete": sum(
            sample["adapter_conditions_complete"]
            for sample in samples
        ),
        "adapter_row_index_correct": sum(
            sample["adapter_row_index_correct"]
            for sample in samples
        ),
    }
    totals["recovered_coverage"] = round(
        totals["recovered_rows_complete"]
        / totals["gold_records"],
        4,
    )
    totals["adapter_coverage"] = round(
        totals["adapter_complete"] / totals["gold_records"],
        4,
    )
    totals["adapter_identity_coverage"] = round(
        totals["adapter_identity_matched"]
        / totals["gold_records"],
        4,
    )
    totals["adapter_value_coverage"] = round(
        totals["adapter_value_present"]
        / totals["gold_records"],
        4,
    )

    payload = {
        "schema": "b35d_structure_compare/v1",
        "note": (
            "结构指标是离线可恢复性，不是真实提取 F1。"
            "Flat 是 B3.5c 实际输入；Recovered/Adapter 未进入 "
            "B3.5c 运行。"
        ),
        "totals": totals,
        "samples": samples,
        "records": all_records,
    }
    output = EXPERIMENT_DIR / "structure/structure_compare.json"
    write_json(output, payload)
    write_csv(
        all_records,
        EXPERIMENT_DIR / "structure/structure_compare.csv",
    )

    print("结构对照:", output)
    print("全局:", totals)

    for sample in samples:
        print(
            " ",
            sample["sample_id"],
            "gold",
            sample["gold_records"],
            "recovered",
            sample["recovered_coverage"],
            "adapter",
            sample["adapter_coverage"],
            "rows",
            sample["adapter_rows_total"],
            "excluded",
            sample["adapter_excluded_rows"],
            sample["adapter_excluded_reasons"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
