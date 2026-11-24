"""B3.6：13 条 Validator 误拒回归（只读，不调用 LLM）。

对 B3.5c 的 artifact.extraction（未校验参数）重新执行：
- verify_single_parameter
- check_min_typ_max
- ParameterValidator

对照 B3.5d 已冻结的复核结果，记录 B3.6 修复后的状态。
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

from app.extraction.evidence import (  # noqa: E402
    verify_single_parameter,
)
from app.validation.rules import check_min_typ_max  # noqa: E402
from app.validation.validator import (  # noqa: E402
    ParameterValidator,
)
from b35d_common import (  # noqa: E402
    RUN_ROOT,
    load_json,
    write_json,
)


EXPERIMENT_DIR = (
    PROJECT_ROOT / "data/outputs/experiments/B3.6"
)
OUTPUT_DIR = EXPERIMENT_DIR / "validator"

B35D_REVIEW = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.5d/validator"
    / "review.json"
)


def selected_tables(sample_id):
    payload = load_json(
        RUN_ROOT / sample_id / "selected_tables.json"
    )

    return {
        table["table_id"]: table
        for table in payload["tables"]
    }


def raw_parameters(sample_id):
    artifact = load_json(
        RUN_ROOT / sample_id / "artifact.json"
    )

    return artifact["extraction"]["parameters"]


def revalidate(params, tables):
    """用 B3.6 生产函数重新执行证据校验 + Validator。"""

    checked = []

    for param in params:
        item = {
            key: value
            for key, value in param.items()
            if key not in (
                "validation_status",
                "validation_errors",
            )
        }
        status, issues = verify_single_parameter(
            item, tables
        )
        item["validation_status"] = status
        item["validation_errors"] = list(issues)
        checked.append(item)

    rule_errors = check_min_typ_max(checked)
    validated = ParameterValidator().validate(
        {"parameters": checked}
    )["parameters"]

    return checked, rule_errors, validated


def main():
    review = load_json(B35D_REVIEW)
    cache = {}
    records = []

    for record in review["records"]:
        sample_id = record["sample_id"]

        if sample_id not in cache:
            params = raw_parameters(sample_id)
            tables = selected_tables(sample_id)
            checked, rule_errors, validated = revalidate(
                params, tables
            )
            cache[sample_id] = {
                "params": checked,
                "rules": rule_errors,
                "validated": validated,
            }

        state = cache[sample_id]
        index = record["prediction_index"]
        checked = state["params"][index]
        final = state["validated"][index]
        before = record["reproduced"]

        records.append({
            "gold_id": record["gold_id"],
            "sample_id": sample_id,
            "prediction_index": index,
            "root_cause": record["root_cause"],
            "root_cause_flags": record["root_cause_flags"],
            "before": {
                "verify_status": before["verify_status"],
                "verify_issues": before["verify_issues"],
                "rule_errors": before["rule_errors"],
                "final_validation_status": before[
                    "final_validation_status"
                ],
                "final_validation_errors": before[
                    "final_validation_errors"
                ],
            },
            "after": {
                "verify_status": checked[
                    "validation_status"
                ],
                "verify_issues": checked[
                    "validation_errors"
                ],
                "rule_errors": state["rules"].get(
                    index, []
                ),
                "final_validation_status": final[
                    "validation_status"
                ],
                "final_validation_errors": final[
                    "validation_errors"
                ],
            },
        })

    before_counts = Counter(
        record["before"]["final_validation_status"]
        for record in records
    )
    after_counts = Counter(
        record["after"]["final_validation_status"]
        for record in records
    )
    by_cause = defaultdict(
        lambda: {
            "before": Counter(),
            "after": Counter(),
        }
    )

    for record in records:
        bucket = by_cause[record["root_cause"]]
        bucket["before"][
            record["before"]["final_validation_status"]
        ] += 1
        bucket["after"][
            record["after"]["final_validation_status"]
        ] += 1

    payload = {
        "schema": "b36_validator_regression/v1",
        "note": (
            "13 条 B3.5d 误拒的 B3.6 复核结果。"
            "只读 B3.5c artifact，不调用 LLM。"
        ),
        "record_count": len(records),
        "before_status_counts": dict(before_counts),
        "after_status_counts": dict(after_counts),
        "by_root_cause": {
            cause: {
                "before": dict(counts["before"]),
                "after": dict(counts["after"]),
            }
            for cause, counts in by_cause.items()
        },
        "records": records,
    }
    write_json(OUTPUT_DIR / "regression.json", payload)

    with (OUTPUT_DIR / "regression.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gold_id",
                "sample_id",
                "root_cause",
                "before_status",
                "after_status",
                "after_verify_issues",
                "after_rule_errors",
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow({
                "gold_id": record["gold_id"],
                "sample_id": record["sample_id"],
                "root_cause": record["root_cause"],
                "before_status": record["before"][
                    "final_validation_status"
                ],
                "after_status": record["after"][
                    "final_validation_status"
                ],
                "after_verify_issues": " | ".join(
                    record["after"]["verify_issues"]
                ),
                "after_rule_errors": " | ".join(
                    record["after"]["rule_errors"]
                ),
            })

    print("B3.6 Validator 回归:", OUTPUT_DIR)
    print("before:", dict(before_counts))
    print("after :", dict(after_counts))

    for record in records:
        if record["after"][
            "final_validation_status"
        ] == "invalid":
            print(
                "  STILL INVALID",
                record["gold_id"],
                record["after"]["final_validation_errors"],
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
