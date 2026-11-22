"""阶段 B3.2b：构建用于 A/B 的 Gold v2 评估标准。

把 supplemented Gold 中的：
- 表级默认条件
- 脚注 (1)～(4)
- 行级 effective conditions / annotations
合并进 Gold v2，生成同一份双 arm 共用的评估文件。

不覆盖 Gold v2、旧 Gold、Excel、PDF 等已有产物。
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
)
DEFAULT_GOLD_V2 = DEFAULT_DIR / "gold_standard.table9.v2.json"
DEFAULT_SUPPLEMENT = (
    DEFAULT_DIR / "gold_standard.table9.supplemented.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_DIR / "gold_standard.table9.v2.eval.json"
)


class EvalGoldError(ValueError):
    """评估 Gold 构建失败。"""


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def key_of(item):
    return (item["logical_row_id"], item["value_type"])


def build_eval_gold(gold_v2, supplement):
    supplement_by_key = {
        key_of(item): item["supplement"]
        for item in supplement["candidate_records"]
    }

    params = []

    for item in gold_v2["expected_parameters"]:
        record = dict(item)
        record["test_conditions_row_level"] = (
            item["test_conditions"]
        )

        supp = supplement_by_key.get(key_of(item))

        if supp is None:
            # UVLO 等 v2 新增行没有脚注；
            # 表级默认条件仍由 eval 文件顶层继承。
            supp = {
                "footnotes": [],
                "annotations": [],
                "effective_conditions": {
                    "row_level_test_conditions": (
                        item["test_conditions"]
                    ),
                    "table_defaults_inherited": True,
                    "note": (
                        "该记录无脚注；A/B 只要求"
                        "行级测试条件精确匹配。"
                    ),
                },
                "condition_changes": [],
                "has_supplement": False,
            }
        else:
            supp = json.loads(
                json.dumps(supp, ensure_ascii=False)
            )

        record["condition_sources"] = {
            "row_level": item["test_conditions"],
            "table_default_applies": True,
            "footnotes": list(
                supp.get("footnotes") or []
            ),
            "supplemental_effective_conditions": (
                supp.get("effective_conditions")
            ),
        }
        record["supplement"] = supp
        params.append(record)

    return {
        "schema": "ab_gold_standard_eval/v2",
        "version": 2,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "table_id": gold_v2["table_id"],
        "source": {
            "gold_v2_path": str(DEFAULT_GOLD_V2),
            "gold_v2_sha256": sha256_file(
                DEFAULT_GOLD_V2
            ),
            "supplement_path": str(
                DEFAULT_SUPPLEMENT
            ),
            "supplement_sha256": sha256_file(
                DEFAULT_SUPPLEMENT
            ),
            "pdf_sha256": (
                gold_v2.get("source") or {}
            ).get("pdf_sha256"),
            "logical_rows_v2_path": (
                gold_v2.get("source") or {}
            ).get("logical_rows_v2_path"),
            "logical_rows_v2_sha256": (
                gold_v2.get("source") or {}
            ).get("logical_rows_v2_sha256"),
        },
        "status_summary": gold_v2[
            "status_summary"
        ],
        "table_default_conditions": supplement[
            "table_default_conditions"
        ],
        "footnotes": supplement["footnotes"],
        "precedence_rules": supplement[
            "precedence_rules"
        ],
        "trusted_logical_row_count": gold_v2[
            "trusted_logical_row_count"
        ],
        "expected_parameter_count": len(params),
        "expected_parameters": params,
        "excluded_needs_review": [],
        "evaluation_rules": {
            "primary_match_key": "logical_row_id + value_type",
            "strict_fields": [
                "identity",
                "value",
                "unit",
                "test_conditions_row_level",
            ],
            "supplemental_conditions_compared_separately": True,
            "footnotes_required_in_llm_output": False,
            "note": (
                "LLM 只要求输出行级 test_conditions；"
                "表级默认条件和脚注作为 Gold 的补充解释，"
                "不要求模型在 extraction 中逐字复述。"
            ),
        },
        "integrity": {
            "record_count": len(params),
            "unique_keys": len({
                key_of(item) for item in params
            }),
            "records_with_supplement": sum(
                1 for item in params
                if item["supplement"].get(
                    "has_supplement"
                )
            ),
            "records_with_footnotes": sum(
                1 for item in params
                if item["supplement"].get("footnotes")
            ),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="构建 Gold v2 A/B 评估标准"
    )
    parser.add_argument(
        "--gold-v2", default=str(DEFAULT_GOLD_V2)
    )
    parser.add_argument(
        "--supplement",
        default=str(DEFAULT_SUPPLEMENT),
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT)
    )
    args = parser.parse_args(argv)

    gold_v2 = load(args.gold_v2)
    supplement = load(args.supplement)

    if gold_v2["expected_parameter_count"] != 51:
        raise EvalGoldError(
            "Gold v2 不是 51 条: "
            f"{gold_v2['expected_parameter_count']}"
        )

    if len(supplement["candidate_records"]) != 44:
        raise EvalGoldError(
            "Supplemented Gold 不是 44 条旧候选"
        )

    result = build_eval_gold(
        gold_v2, supplement
    )

    if result["integrity"]["record_count"] != 51:
        raise EvalGoldError("评估 Gold 记录数不是 51")

    if (
        result["integrity"]["unique_keys"]
        != result["integrity"]["record_count"]
    ):
        raise EvalGoldError("评估 Gold 存在重复键")

    output = Path(args.output)
    output.write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 68)
    print("Gold v2 A/B 评估标准已生成")
    print("=" * 68)
    print("输出:", output)
    print("记录:", result["expected_parameter_count"])
    print("唯一键:", result["integrity"]["unique_keys"])
    print(
        "含脚注补充:",
        result["integrity"]["records_with_footnotes"],
    )
    print(
        "表级默认:",
        result["table_default_conditions"]["text"],
    )
    print(
        "脚注:",
        ", ".join(
            f"({number})" for number in sorted(
                result["footnotes"]
            )
        ),
    )
    print(
        "PDF SHA256:",
        result["source"]["pdf_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
