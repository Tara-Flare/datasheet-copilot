"""B3.6：修复后 StructureRecovery / LogicalTableAdapter 离线对照。

只读 B3.5d 的 raw Docling 结构，重新跑 recover_from_pdf，
不调用 LLM、不改 B3.5b Gold、不覆盖 B3.5c/B3.5d 产物。

输出的结构指标只描述“离线可恢复性/可适配率”，
不是真实提取 Recall/F1。真实效果必须等受控 A/B。
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for path in (SCRIPT_DIR, PROJECT_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.extraction.logical_table_adapter import (  # noqa: E402
    adapt_logical_rows,
)
from app.parsers.structure_recovery import (  # noqa: E402
    recover_from_pdf,
)
from b35c_score_flat import normalize_gold  # noqa: E402
from b35d_structure_compare import (  # noqa: E402
    TARGETS,
    analyze_record,
    flat_rows_for_gold_table,
    summarize,
    write_csv,
)
from b35d_common import (  # noqa: E402
    load_gold,
    load_json,
    load_raw_structure,
    write_json,
)


EXPERIMENT_DIR = (
    PROJECT_ROOT / "data/outputs/experiments/B3.6"
)
STRUCTURE_DIR = EXPERIMENT_DIR / "structure"

B35D_STRUCTURE = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/structure"
    / "structure_compare.json"
)

PDF_DIR = PROJECT_ROOT / "data/samples/b35b"

# 1-based PDF 页码，与 selected_tables.page_numbers 一致。
PDF_PAGES = {
    "adi_lt3045": 4,
    "diodes_ap2112": 4,
    "onsemi_ncp170": 3,
    "renesas_isl80510": 4,
}


def recover_rows(sample_id):
    raw = load_raw_structure(sample_id)

    if raw is None:
        raise SystemExit(f"缺少 {sample_id} 的 raw 结构")

    pdf = PDF_DIR / f"{sample_id}.pdf"

    if not pdf.is_file():
        raise SystemExit(f"缺少 PDF: {pdf}")

    return recover_from_pdf(
        raw,
        TARGETS[sample_id],
        str(pdf),
        PDF_PAGES[sample_id],
    )


def build_adapter(sample_id, recovered, raw):
    return adapt_logical_rows(
        recovered,
        raw_structure=raw,
        table_ids=[TARGETS[sample_id]],
    )


def write_logical(sample_id, recovered):
    path = (
        STRUCTURE_DIR
        / f"{sample_id}.logical_rows.v1.json"
    )
    write_json(path, recovered)
    return path


def before_after(before, after):
    keys = [
        "gold_records",
        "recovered_rows_complete",
        "recovered_coverage",
        "adapter_rows_matched",
        "adapter_identity_matched",
        "adapter_value_present",
        "adapter_unit_present",
        "adapter_conditions_complete",
        "adapter_row_index_correct",
        "adapter_complete",
        "adapter_coverage",
    ]
    return {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "delta": (
                round(after.get(key) - before.get(key), 4)
                if isinstance(before.get(key), (int, float))
                and isinstance(after.get(key), (int, float))
                else None
            ),
        }
        for key in keys
    }


def adi_table4_rows(records):
    """ADI table 4 每条 Gold 的前后结构状态。"""

    out = []

    for record in records:
        if record["sample_id"] != "adi_lt3045":
            continue

        recovered = record["structure_recovery"]["rows"]
        adapter = record["adapter"]["rows"]
        out.append({
            "gold_id": record["gold_id"],
            "value_type": record["value_type"],
            "value": record["value"],
            "unit": record["unit"],
            "recovered_row_found": record[
                "structure_recovery"
            ]["row_found"],
            "recovered_complete": record[
                "structure_recovery"
            ]["any_complete"],
            "recovered_logical_row_ids": [
                row["logical_row_id"]
                for row in recovered
            ],
            "recovered_values": [
                {
                    "logical_row_id": row["logical_row_id"],
                    "min": row["min"],
                    "typ": row["typ"],
                    "max": row["max"],
                    "unit": row["unit"],
                }
                for row in recovered
            ],
            "adapter_row_found": record["adapter"][
                "row_found"
            ],
            "adapter_complete": record["adapter"][
                "any_complete"
            ],
            "adapter_logical_row_ids": [
                row["logical_row_id"]
                for row in adapter
            ],
        })

    return out


def main():
    gold = load_gold()
    before = load_json(B35D_STRUCTURE)
    before_by_sample = {
        sample["sample_id"]: sample
        for sample in before["samples"]
    }

    all_records = []
    samples = []

    for sample in gold["samples"]:
        sample_id = sample["sample_id"]
        raw = load_raw_structure(sample_id)
        recovered = recover_rows(sample_id)
        adapted = build_adapter(
            sample_id, recovered, raw
        )
        write_logical(sample_id, recovered)

        records = [
            analyze_record(record, recovered, adapted)
            for record in sample["records"]
        ]
        all_records.extend(records)
        summary = summarize(sample, records)
        table = adapted["tables"][0]
        summary.update({
            "table_id": TARGETS[sample_id],
            "logical_rows_total": len(
                recovered.get("logical_rows", [])
            ),
            "logical_needs_review": sum(
                1
                for row in recovered.get(
                    "logical_rows", []
                )
                if row.get("needs_review")
            ),
            "unresolved_issue_count": len(
                recovered.get(
                    "unresolved_issues", []
                )
            ),
            "unresolved_issues": recovered.get(
                "unresolved_issues", []
            ),
            "adapter_rows_total": table.get("row_count"),
            "adapter_needs_review_excluded": len(
                table.get("excluded_rows", [])
            ),
            "adapter_excluded_rows": table.get(
                "excluded_rows", []
            ),
            "flat_table": flat_rows_for_gold_table(
                sample_id,
                [
                    normalize_gold(record)
                    for record in sample["records"]
                ],
            ),
            "before_after": before_after(
                before_by_sample[sample_id],
                summarize(sample, records),
            ),
        })
        samples.append(summary)

    before_totals = before["totals"]
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
        "schema": "b36_structure_compare/v1",
        "note": (
            "结构指标是离线可恢复性，不是真实提取 F1。"
            "Flat 是 B3.5c 实际输入；B3.6 Recovered/Adapter "
            "尚未进入任何 LLM 运行。"
        ),
        "totals": totals,
        "before_totals": before_totals,
        "totals_before_after": before_after(
            before_totals, totals
        ),
        "samples": samples,
        "records": all_records,
    }
    write_json(
        STRUCTURE_DIR / "structure_compare.json",
        payload,
    )
    write_csv(
        all_records,
        STRUCTURE_DIR / "structure_compare.csv",
    )
    write_json(
        STRUCTURE_DIR / "adi_table4_before_after.json",
        adi_table4_rows(all_records),
    )

    print("B3.6 结构对照:", STRUCTURE_DIR)
    print("全局 before:", before_totals)
    print("全局 after :", totals)

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
            "review",
            sample["logical_needs_review"],
            "issues",
            sample["unresolved_issue_count"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
