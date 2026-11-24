"""B3.5d：B3.5c Flat Baseline 调用/Token 成本拆解（只读）。

回答两个问题：
1. 50 条 FN 的调用成本主要花在 Gold scope 表还是无关表；
2. 输出截断（finish_reason=length）消耗了多少 completion token。
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

from b35d_common import (  # noqa: E402
    EXPERIMENT_DIR,
    attach_events,
    load_gold,
    write_json,
)


OUTPUT_DIR = EXPERIMENT_DIR / "cost"


def empty_bucket():
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "length_calls": 0,
        "empty_content_calls": 0,
        "parameters": 0,
    }


def add(bucket, event):
    bucket["calls"] += 1
    bucket["prompt_tokens"] += event["prompt_tokens"]
    bucket["completion_tokens"] += event["completion_tokens"]
    bucket["reasoning_tokens"] += event["reasoning_tokens"]
    bucket["length_calls"] += int(
        event["finish"] == "length"
    )
    bucket["empty_content_calls"] += int(
        event["content_length"] == 0
    )
    bucket["parameters"] += len(event.get("parameters", []))


def build():
    gold = load_gold()
    gold_tables = defaultdict(set)

    for sample in gold["samples"]:
        for record in sample["records"]:
            gold_tables[sample["sample_id"]].add(
                record["source"]["table_id"]
            )

    samples = []
    total = empty_bucket()
    total_extract = empty_bucket()
    total_retry = empty_bucket()
    total_gold_table = empty_bucket()
    total_non_gold_table = empty_bucket()
    total_length = empty_bucket()
    table_rows = []

    for sample in gold["samples"]:
        sample_id = sample["sample_id"]
        events = attach_events(sample_id)
        scope = gold_tables[sample_id]
        bucket = empty_bucket()
        extract = empty_bucket()
        retry = empty_bucket()
        gold_table = empty_bucket()
        non_gold_table = empty_bucket()
        length = empty_bucket()
        per_table = defaultdict(empty_bucket)

        for event in events:
            add(bucket, event)

            if event["kind"] == "extract":
                add(extract, event)

                if event["table_id"] in scope:
                    add(gold_table, event)
                else:
                    add(non_gold_table, event)

                add(
                    per_table[event["table_id"]], event
                )
            else:
                add(retry, event)

                for table_id in event.get("table_ids", []):
                    add(
                        per_table[table_id],
                        {
                            **event,
                            "content_length": 0,
                        },
                    )

            if event["finish"] == "length":
                add(length, event)

        for bucket_value, target in (
            (bucket, total),
            (extract, total_extract),
            (retry, total_retry),
            (gold_table, total_gold_table),
            (non_gold_table, total_non_gold_table),
            (length, total_length),
        ):
            for key, value in bucket_value.items():
                target[key] += value

        tables = []

        for table_id, table_bucket in sorted(
            per_table.items(), key=lambda item: -item[1]["calls"]
        ):
            tables.append(
                {
                    "table_id": table_id,
                    "is_gold_scope": table_id in scope,
                    **table_bucket,
                }
            )
            table_rows.append(
                {
                    "sample_id": sample_id,
                    "table_id": table_id,
                    "is_gold_scope": table_id in scope,
                    **table_bucket,
                }
            )

        samples.append(
            {
                "sample_id": sample_id,
                "gold_tables": sorted(scope),
                "selected_table_ids": sorted(
                    per_table
                ),
                "total": bucket,
                "extract": extract,
                "retry": retry,
                "gold_table_extract": gold_table,
                "non_gold_table_extract": non_gold_table,
                "length_calls": length,
                "per_table": tables,
                "non_gold_table_call_ratio": round(
                    non_gold_table["calls"]
                    / extract["calls"],
                    4,
                )
                if extract["calls"]
                else None,
                "retry_call_ratio": round(
                    retry["calls"] / bucket["calls"], 4
                )
                if bucket["calls"]
                else None,
                "length_call_ratio": round(
                    length["calls"] / bucket["calls"], 4
                )
                if bucket["calls"]
                else None,
            }
        )

    totals = {
        "calls": total["calls"],
        "prompt_tokens": total["prompt_tokens"],
        "completion_tokens": total["completion_tokens"],
        "reasoning_tokens": total["reasoning_tokens"],
        "total_tokens": (
            total["prompt_tokens"]
            + total["completion_tokens"]
        ),
        "extract": total_extract,
        "retry": total_retry,
        "gold_table_extract": total_gold_table,
        "non_gold_table_extract": total_non_gold_table,
        "length_calls": total_length,
        "non_gold_table_call_ratio": round(
            total_non_gold_table["calls"]
            / total_extract["calls"],
            4,
        )
        if total_extract["calls"]
        else None,
        "retry_call_ratio": round(
            total_retry["calls"] / total["calls"], 4
        )
        if total["calls"]
        else None,
        "length_call_ratio": round(
            total_length["calls"] / total["calls"], 4
        )
        if total["calls"]
        else None,
        "length_completion_token_share": round(
            total_length["completion_tokens"]
            / total["completion_tokens"],
            4,
        )
        if total["completion_tokens"]
        else None,
        "non_gold_table_completion_token_share": round(
            total_non_gold_table["completion_tokens"]
            / total["completion_tokens"],
            4,
        )
        if total["completion_tokens"]
        else None,
    }

    return {
        "schema": "b35d_cost_breakdown/v1",
        "note": (
            "retry 调用无法唯一拆分到单表；per_table 中 retry token "
            "按 table_ids 复制计入，因此表级 token 之和会大于全局。"
            "全局 retry/total 不重复计数。"
        ),
        "totals": totals,
        "samples": samples,
        "per_table_rows": table_rows,
    }


def write_csv(rows, path):
    fieldnames = [
        "sample_id",
        "table_id",
        "is_gold_scope",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "length_calls",
        "empty_content_calls",
        "parameters",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    payload = build()
    write_json(OUTPUT_DIR / "cost_breakdown.json", payload)
    write_csv(
        payload["per_table_rows"],
        OUTPUT_DIR / "cost_breakdown.csv",
    )

    totals = payload["totals"]
    print("总调用:", totals["calls"])
    print(
        "token: prompt",
        totals["prompt_tokens"],
        "completion",
        totals["completion_tokens"],
        "reasoning",
        totals["reasoning_tokens"],
    )
    print(
        "非 Gold scope 表 extract 调用占比:",
        totals["non_gold_table_call_ratio"],
    )
    print(
        "retry 调用占比:", totals["retry_call_ratio"]
    )
    print(
        "length 调用占比:", totals["length_call_ratio"],
        "completion token 占比:",
        totals["length_completion_token_share"],
    )

    for sample in payload["samples"]:
        print(
            " ",
            sample["sample_id"],
            "calls",
            sample["total"]["calls"],
            "extract",
            sample["extract"]["calls"],
            "retry",
            sample["retry"]["calls"],
            "non-gold",
            sample["non_gold_table_extract"]["calls"],
            "length",
            sample["length_calls"]["calls"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
