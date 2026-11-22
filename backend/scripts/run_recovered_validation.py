"""阶段 B2.1.1：运行离线证据匹配并保存报告。

只读：不调用 DeepSeek，不重新解析 PDF，不修改原始提取结果。
"""

import argparse
import json
from pathlib import Path

from app.validation.recovered_evidence import (
    validate_from_files,
)


def compare_to_previous(report, previous):
    previous_by_index = {
        item["index"]: item
        for item in previous.get("parameters", [])
    }

    counts = {
        "downgraded_validated_to_correctable": 0,
        "downgraded_validated_to_invalid": 0,
        "downgraded_validated_to_pending": 0,
        "newly_validated": 0,
        "other_changes": 0,
    }

    changes = []

    for item in report["parameters"]:
        before_item = previous_by_index.get(item["index"])

        if before_item is None:
            continue

        before = before_item.get("recovered_status")
        after = item.get("recovered_status")

        if before == after:
            continue

        if before == "validated" and after == "correctable":
            change_type = "validated_to_correctable"
            counts["downgraded_validated_to_correctable"] += 1
        elif before == "validated" and after == "invalid":
            change_type = "validated_to_invalid"
            counts["downgraded_validated_to_invalid"] += 1
        elif before == "validated" and after == "pending_review":
            change_type = "validated_to_pending"
            counts["downgraded_validated_to_pending"] += 1
        elif before != "validated" and after == "validated":
            change_type = "newly_validated"
            counts["newly_validated"] += 1
        else:
            change_type = "other"
            counts["other_changes"] += 1

        changes.append({
            "index": item["index"],
            "name": item.get("name"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "value_type": item.get("value_type"),
            "before_status": before,
            "after_status": after,
            "change_type": change_type,
            "reason": item.get("reason"),
            "conflicts": item.get("conflicts", []),
            "missing_fields": item.get("missing_fields", []),
            "missing_conditions": item.get(
                "field_checks", {}
            ).get("test_conditions", {}).get(
                "missing_conditions", []
            ),
            "selected_logical_row_id": item.get(
                "selected_logical_row_id"
            ),
        })

    return {
        "previous_summary": previous.get("summary"),
        "current_summary": report.get("summary"),
        "counts": counts,
        "changes": changes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="原始参数与逻辑行的离线证据匹配"
    )
    parser.add_argument(
        "--extraction",
        default="data/outputs/TPS7A02.extraction.json",
    )
    parser.add_argument(
        "--logical-rows",
        default="data/outputs/TPS7A02.logical_rows.v1.json",
    )
    parser.add_argument(
        "--baseline",
        default="data/outputs/TPS7A02.final.validation.json",
    )
    parser.add_argument(
        "--raw-structure",
        default="data/outputs/TPS7A02.raw_structure.v1.json",
    )
    parser.add_argument(
        "--previous",
        default=None,
        help="上一版（B2.1）报告，用于逐参数对比",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/outputs/"
            "TPS7A02.recovered_validation.v1.json"
        ),
    )

    args = parser.parse_args(argv)

    extraction = json.loads(
        Path(args.extraction).read_text(encoding="utf-8")
    )
    logical_rows = json.loads(
        Path(args.logical_rows).read_text(encoding="utf-8")
    )
    baseline = json.loads(
        Path(args.baseline).read_text(encoding="utf-8")
    )
    raw_structure = json.loads(
        Path(args.raw_structure).read_text(encoding="utf-8")
    )

    report = validate_from_files(
        extraction,
        logical_rows,
        baseline=baseline,
        raw_structure=raw_structure,
    )

    if args.previous:
        previous = json.loads(
            Path(args.previous).read_text(encoding="utf-8")
        )
        report["comparison_to_previous"] = compare_to_previous(
            report, previous
        )

    output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(output)

    print("恢复校验报告已保存:", output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if "comparison_to_previous" in report:
        print(
            "comparison counts:",
            json.dumps(
                report["comparison_to_previous"]["counts"],
                ensure_ascii=False,
            ),
        )

    print(
        "potential_new_extractable:",
        len(report["potential_new_extractable"]),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
