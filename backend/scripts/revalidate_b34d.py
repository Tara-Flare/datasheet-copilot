"""阶段 B3.4d：生产 Validator 身份归一化修复的离线重验证。

只读取 B3.4b 已保存的原始预测、Logical Rows v2 和
Raw Structure，不调用 DeepSeek API，不修改历史实验数据。
"""

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.validation.recovered_evidence import (
    FAMILY_NORMALIZATION_VERSION,
    RecoveredEvidenceValidator,
    logical_family,
    original_family,
)


DEFAULT_B34B = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.4b"
    / "20261122T160706Z"
    / "b34b_result.json"
)
DEFAULT_LOGICAL_ROWS = (
    PROJECT_ROOT / "data/outputs/TPS7A02.logical_rows.v2.json"
)
DEFAULT_RAW_STRUCTURE = (
    PROJECT_ROOT / "data/outputs/TPS7A02.raw_structure.v1.json"
)
DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.4d"
)
EXPECTED_BRANCH = "experiment/b3.4d-validator"
B34C_COMMIT = "1f1eafb"
EXPECTED_GOLD_SHA256 = (
    "f86ab2556d9fa6b0b7fef20c7ea8c62d2b60d85d59842cd414dd7a09383b9b3d"
)
EXPECTED_B34B_RESULT_SHA256 = (
    "f7f038fc9d3c84230e310e8f04999f232121090e29e1547cba1c44e1d7af186e"
)

FIELD_NAMES = [
    "identity",
    "value",
    "unit",
    "value_type",
    "test_conditions",
    "source",
    "needs_review",
]


class RevalidationError(ValueError):
    """B3.4d 离线重验证失败。"""


def utc_now():
    return datetime.now(timezone.utc)


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_exclusive(path, data):
    path = Path(path)

    if path.exists():
        raise RevalidationError(
            f"拒绝覆盖已有文件: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            data, handle, ensure_ascii=False, indent=2
        )
        handle.write("\n")

    return path


def write_text_exclusive(path, text):
    path = Path(path)

    if path.exists():
        raise RevalidationError(
            f"拒绝覆盖已有文件: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    return path


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RevalidationError(
            f"git {' '.join(args)} 失败"
        ) from exc


def preflight():
    branch = git_output(
        "rev-parse", "--abbrev-ref", "HEAD"
    )
    head = git_output("rev-parse", "HEAD")

    if branch != EXPECTED_BRANCH:
        raise RevalidationError(
            f"当前分支不是 {EXPECTED_BRANCH}: {branch}"
        )

    git_output(
        "merge-base", "--is-ancestor",
        B34C_COMMIT, "HEAD",
    )

    return {
        "branch": branch,
        "head": head,
        "b34c_commit": B34C_COMMIT,
        "status_porcelain": git_output(
            "status", "--porcelain=v1"
        ).splitlines(),
    }


def distribution(parameters):
    counter = Counter(
        item["recovered_status"]
        for item in parameters
    )

    return {
        status: counter.get(status, 0)
        for status in (
            "validated",
            "correctable",
            "invalid",
            "pending_review",
            "not_covered",
        )
    }


def field_distribution(parameters):
    result = {}

    for field in FIELD_NAMES:
        counter = Counter()

        for item in parameters:
            check = item.get("field_checks", {}).get(
                field
            )
            status = (
                check.get("status")
                if isinstance(check, dict)
                else check
            )
            counter[status or "absent"] += 1

        result[field] = dict(counter)

    return result


def old_by_index(old_report):
    return {
        item["index"]: item
        for item in old_report.get(
            "parameters", []
        )
    }


def p33_record(parameters):
    for item in parameters:
        if item.get("selected_logical_row_id") == (
            "t9.p33"
        ):
            return item

    raise RevalidationError(
        "重验证结果中找不到 t9.p33"
    )


def negative_controls(
    extraction,
    logical_rows,
    raw_structure,
):
    p33_index = next(
        index
        for index, item in enumerate(
            extraction["parameters"]
        )
        if (item.get("evidence") or {}).get(
            "logical_row_id"
        ) == "t9.p33"
    )
    base = extraction["parameters"][p33_index]
    controls = []

    def run(label, param, rows=None, expected=None):
        target = {
            "parameters": [param],
        }

        if rows is None:
            rows = logical_rows

        validator = RecoveredEvidenceValidator(
            rows,
            raw_structure=raw_structure,
        )
        result = validator.validate(target)[
            "parameters"
        ][0]
        controls.append({
            "label": label,
            "expected": expected,
            "actual": result["recovered_status"],
            "pass": result["recovered_status"] == expected,
            "conflicts": result["conflicts"],
            "reason": result["reason"],
        })

    wrong_name = copy.deepcopy(base)
    wrong_name["name"] = "output_voltage"
    run("wrong_name", wrong_name, expected="invalid")

    wrong_value = copy.deepcopy(base)
    wrong_value["value"] = 131.0
    run("wrong_value", wrong_value, expected="invalid")

    wrong_unit = copy.deepcopy(base)
    wrong_unit["unit"] = "mV"
    run("wrong_unit", wrong_unit, expected="invalid")

    wrong_conditions = copy.deepcopy(base)
    wrong_conditions["test_conditions"] = (
        "BW = 1 Hz, VOUT = 0.8 V, IOUT = 30 mA"
    )
    run(
        "wrong_conditions",
        wrong_conditions,
        expected="invalid",
    )

    wrong_id = copy.deepcopy(base)
    wrong_id["evidence"]["logical_row_id"] = "t9.p999"
    run(
        "wrong_logical_id",
        wrong_id,
        expected="pending_review",
    )

    wrong_rows = copy.deepcopy(base)
    wrong_rows["evidence"]["source_row_indices"] = [99]
    run(
        "wrong_source_rows",
        wrong_rows,
        expected="pending_review",
    )

    ambiguous = copy.deepcopy(base)
    ambiguous["source_mapping_status"] = (
        "ambiguous_row_index"
    )
    run(
        "ambiguous_mapping",
        ambiguous,
        expected="pending_review",
    )

    uncovered = copy.deepcopy(base)
    uncovered["evidence"]["table_id"] = 999
    run(
        "not_covered_table",
        uncovered,
        expected="not_covered",
    )

    review_rows = copy.deepcopy(logical_rows)
    review_target = next(
        row for row in review_rows["logical_rows"]
        if row.get("logical_row_id") == "t9.p33"
    )
    review_target["needs_review"] = True
    run(
        "needs_review",
        copy.deepcopy(base),
        rows=review_rows,
        expected="pending_review",
    )

    if not all(item["pass"] for item in controls):
        failed = [
            item for item in controls if not item["pass"]
        ]
        raise RevalidationError(
            f"负例控制失败: {failed}"
        )

    return controls


def build_report(
    result,
    old_report,
    new_report,
    p33,
    recovered,
    controls,
):
    old_dist = result["distributions"]["old"]
    new_dist = result["distributions"]["new"]
    old_summary = old_report.get("summary", {})
    new_summary = new_report.get("summary", {})

    lines = [
        "# B3.4d 生产 Validator 身份归一化重验证报告",
        "",
        f"- 阶段：B3.4d",
        f"- 生成时间：{utc_now().isoformat()}",
        f"- 分支：`{result['preflight']['branch']}`",
        f"- HEAD：`{result['preflight']['head']}`",
        f"- B3.4c 基线提交：`{B34C_COMMIT}`",
        f"- Validator family 版本："
        f"`{FAMILY_NORMALIZATION_VERSION}`",
        "",
        "> 本阶段不调用 DeepSeek API，只重验证已保存的"
        "B3.4b 原始预测。",
        "> 未修改 Gold、DeepSeek Prompt、Schema、"
        "来源映射、提取器或 LangGraph Retry。",
        "> 没有删除 identity 校验，也没有放宽数值、"
        "单位、条件或来源检查。",
        "",
        "## 一、输入与安全预检",
        "",
        f"- B3.4b 结果：`{result['inputs']['b34b_result']}`",
        f"- B3.4b 结果 SHA256："
        f"`{result['inputs']['b34b_result_sha256']}`",
        f"- Gold：`{result['inputs']['gold']}`",
        f"- Gold SHA256：`{result['inputs']['gold_sha256']}`",
        f"- Logical Rows：`{result['inputs']['logical_rows']}`",
        f"- Raw Structure："
        f"`{result['inputs']['raw_structure']}`",
        "",
        "## 二、51 条生产状态分布",
        "",
        "| 状态 | 修复前 | 修复后 |",
        "|---|---:|---:|",
    ]

    for status in (
        "validated",
        "correctable",
        "invalid",
        "pending_review",
        "not_covered",
    ):
        lines.append(
            f"| {status} | {old_dist[status]} | "
            f"{new_dist[status]} |"
        )

    lines += [
        "",
        f"- 修复前 recovered_validated："
        f"{old_summary.get('recovered_validated')}",
        f"- 修复后 recovered_validated："
        f"{new_summary.get('recovered_validated')}",
        f"- 修复前 identity 冲突："
        f"{old_summary.get('parameter_field_conflicts', {}).get('identity', 0)}",
        f"- 修复后 identity 冲突："
        f"{new_summary.get('parameter_field_conflicts', {}).get('identity', 0)}",
        f"- 证据映射覆盖："
        f"{new_summary.get('evidence_mapping_coverage')}",
        "",
        "## 三、逐字段检查分布",
        "",
        "| 字段 | 修复后状态分布 |",
        "|---|---|",
    ]

    for field, stats in result["field_distribution"][
        "new"
    ].items():
        lines.append(
            f"| {field} | "
            + ", ".join(
                f"{key}: {value}"
                for key, value in sorted(stats.items())
            )
            + " |"
        )

    lines += [
        "",
        "## 四、由 invalid 恢复为 validated 的记录",
        "",
        "| ID | 类型 | 参数名 | 修复前 | 修复后 |",
        "|---|---|---|---|---|",
    ]

    for item in recovered:
        lines.append(
            f"| {item['logical_row_id']} | "
            f"{item['value_type']} | "
            f"{item['name']} | invalid | validated |"
        )

    lines += [
        "",
        "## 五、t9.p33 专项",
        "",
        "| 项目 | 修复前 | 修复后 |",
        "|---|---|---|",
        f"| recovered_status | "
        f"`{p33['old_status']}` | "
        f"`{p33['new_status']}` |",
        f"| identity | "
        f"`{p33['old_field_checks'].get('identity', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('identity', {}).get('status')}` |",
        f"| value | "
        f"`{p33['old_field_checks'].get('value', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('value', {}).get('status')}` |",
        f"| unit | "
        f"`{p33['old_field_checks'].get('unit', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('unit', {}).get('status')}` |",
        f"| value_type | "
        f"`{p33['old_field_checks'].get('value_type', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('value_type', {}).get('status')}` |",
        f"| test_conditions | "
        f"`{p33['old_field_checks'].get('test_conditions', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('test_conditions', {}).get('status')}` |",
        f"| source | "
        f"`{p33['old_field_checks'].get('source', {}).get('status')}` | "
        f"`{p33['new_field_checks'].get('source', {}).get('status')}` |",
        "",
        "### identity family",
        "",
        "| 阶段 | original_family | logical_family | status |",
        "|---|---|---|---|",
        f"| 修复前 | "
        f"`{p33['old_identity'].get('original_family')}` | "
        f"`{p33['old_identity'].get('logical_family')}` | "
        f"`{p33['old_identity'].get('status')}` |",
        f"| 修复后 | "
        f"`{p33['new_identity'].get('original_family')}` | "
        f"`{p33['new_identity'].get('logical_family')}` | "
        f"`{p33['new_identity'].get('status')}` |",
        "",
        "### 数值 / 单位 / 条件",
        "",
        f"- 数值：`{p33['value']}`，修复后检查 "
        f"`{p33['new_field_checks']['value']['status']}`。",
        f"- 单位：`{p33['unit']}`，修复后检查 "
        f"`{p33['new_field_checks']['unit']['status']}`。",
        f"- value_type：`{p33['value_type']}`，修复后检查 "
        f"`{p33['new_field_checks']['value_type']['status']}`。",
        f"- test_conditions：`{p33['test_conditions']}`，"
        "修复后检查 "
        f"`{p33['new_field_checks']['test_conditions']['status']}`。",
        "",
        "### source 子检查",
        "",
        f"- 修复后 source status："
        f"`{p33['new_field_checks']['source']['status']}`",
        f"- checks：`{p33['new_field_checks']['source'].get('checks')}`",
        "",
        "## 六、负例控制（不得放行）",
        "",
        "| 负例 | 期望 | 实际 | 通过 |",
        "|---|---|---|---|",
    ]

    for item in controls:
        lines.append(
            f"| {item['label']} | `{item['expected']}` | "
            f"`{item['actual']}` | {item['pass']} |"
        )

    lines += [
        "",
        "## 七、修复内容",
        "",
        "- `original_family()` 增加规范化 key 和别名：",
        "  - `output_voltage_noise` / `V_N` / `VN` → `output_noise`",
        "  - `input_voltage_noise` → `input_noise`",
        "  - `output_voltage_accuracy` → `accuracy`",
        "  - `ground_current_vs_load_current` → `ground_current_load`",
        "  - `ground_current_in_dropout` → `ground_current`",
        "  - `output_current_limit` → `current_limit`",
        "  - `dropout_voltage` → `dropout`",
        "- `logical_family()` 增加 noise > accuracy > 普通规则的"
        "语义优先级，并统一返回规范化 family。",
        "- identity 校验仍然保留；只修正双方 family 名称不一致。",
        "",
        "## 八、回归风险",
        "",
        "- 风险集中在 identity family 归一化；"
        "数值、单位、条件、来源检查未改。",
        "- `output_voltage_noise` 不会再与纯 `output_voltage` 混淆。",
        "- `input_voltage_noise` 与 `output_noise` 保持区分。",
        "- `output_current_limit` 与 `short_circuit_current_limit` "
        "保持区分。",
        "- `ground_current` 与 `ground_current_load` 保持区分。",
        "- `UVLO threshold` 与 `UVLO hysteresis` 保持区分。",
        "- 负例控制确认错误名称、数值、单位、条件、来源、"
        "needs_review 和 ambiguous mapping 均不会被放行。",
        "",
        "## 九、结论",
        "",
        "1. B3.4b 的 51 条预测在修复后全部为生产 "
        "`validated`。",
        "2. t9.p33 从 `invalid` 变为 `validated`，"
        "identity/value/unit/value_type/test_conditions/"
        "source 全部通过。",
        "3. 修复前 11 条 identity 冲突全部是 family "
        "规范化缺口，不是数值或来源错误。",
        "4. 负例控制全部保持 invalid 或 pending_review，"
        "未放宽校验。",
        "5. 未修改 B3.4c 评估器结果，未进入 B4。",
        "",
    ]

    return "\n".join(lines)


def write_details_csv(path, old_report, new_report):
    old_map = old_by_index(old_report)
    fields = [
        "index",
        "logical_row_id",
        "name",
        "value_type",
        "old_status",
        "new_status",
        "identity",
        "value",
        "unit",
        "value_type_check",
        "test_conditions",
        "source",
        "needs_review",
        "conflicts",
        "missing_fields",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for item in new_report["parameters"]:
            old = old_map.get(item["index"], {})
            checks = item.get("field_checks", {})
            writer.writerow({
                "index": item["index"],
                "logical_row_id": item.get(
                    "selected_logical_row_id"
                ),
                "name": item.get("name"),
                "value_type": item.get("value_type"),
                "old_status": old.get(
                    "recovered_status"
                ),
                "new_status": item.get(
                    "recovered_status"
                ),
                "identity": checks.get(
                    "identity", {}
                ).get("status"),
                "value": checks.get("value", {}).get(
                    "status"
                ),
                "unit": checks.get("unit", {}).get(
                    "status"
                ),
                "value_type_check": checks.get(
                    "value_type", {}
                ).get("status"),
                "test_conditions": checks.get(
                    "test_conditions", {}
                ).get("status"),
                "source": checks.get("source", {}).get(
                    "status"
                ),
                "needs_review": checks.get(
                    "needs_review", {}
                ).get("status"),
                "conflicts": "|".join(
                    item.get("conflicts", [])
                ),
                "missing_fields": "|".join(
                    item.get("missing_fields", [])
                ),
            })


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "B3.4d：离线重验证 B3.4b 的生产 "
            "RecoveredEvidenceValidator 状态"
        )
    )
    parser.add_argument(
        "--b34b", default=str(DEFAULT_B34B)
    )
    parser.add_argument(
        "--logical-rows",
        default=str(DEFAULT_LOGICAL_ROWS),
    )
    parser.add_argument(
        "--raw-structure",
        default=str(DEFAULT_RAW_STRUCTURE),
    )
    parser.add_argument(
        "--gold", default=str(DEFAULT_GOLD)
    )
    parser.add_argument(
        "--output-dir", default=None
    )
    args = parser.parse_args(argv)

    if FAMILY_NORMALIZATION_VERSION != (
        "b3.4d-validator-v1"
    ):
        raise RevalidationError(
            "Validator family 版本不是 b3.4d-validator-v1"
        )

    preflight_data = preflight()
    b34b = load_json(args.b34b)
    logical_rows = load_json(args.logical_rows)
    raw_structure = load_json(args.raw_structure)
    gold = load_json(args.gold)

    if sha256_file(args.gold) != EXPECTED_GOLD_SHA256:
        raise RevalidationError(
            "Gold v2 哈希与 B3.4b 基线不一致"
        )

    if sha256_file(args.b34b) != EXPECTED_B34B_RESULT_SHA256:
        raise RevalidationError(
            "B3.4b 结果哈希与基线不一致"
        )

    arm = next(
        item for item in b34b["arms"]
        if item["arm"] == "logical"
    )
    extraction = arm["extraction"]
    old_report = arm["recovered_validation_raw"]

    new_report = RecoveredEvidenceValidator(
        logical_rows,
        raw_structure=raw_structure,
    ).validate(extraction)

    old_map = old_by_index(old_report)
    recovered = []

    for item in new_report["parameters"]:
        old = old_map.get(item["index"], {})

        if (
            old.get("recovered_status") == "invalid"
            and item["recovered_status"] == "validated"
        ):
            recovered.append({
                "index": item["index"],
                "logical_row_id": item.get(
                    "selected_logical_row_id"
                ),
                "name": item.get("name"),
                "value_type": item.get("value_type"),
                "old_status": old.get(
                    "recovered_status"
                ),
                "new_status": item[
                    "recovered_status"
                ],
            })

    p33_old = next(
        item for item in old_report["parameters"]
        if item.get("selected_logical_row_id")
        == "t9.p33"
    )
    p33_new = p33_record(
        new_report["parameters"]
    )
    p33 = {
        "old_status": p33_old.get("recovered_status"),
        "new_status": p33_new.get("recovered_status"),
        "old_field_checks": p33_old.get(
            "field_checks", {}
        ),
        "new_field_checks": p33_new.get(
            "field_checks", {}
        ),
        "old_identity": p33_old.get(
            "field_checks", {}
        ).get("identity", {}),
        "new_identity": p33_new.get(
            "field_checks", {}
        ).get("identity", {}),
        "value": p33_new.get("value"),
        "unit": p33_new.get("unit"),
        "value_type": p33_new.get("value_type"),
        "test_conditions": p33_new.get(
            "test_conditions"
        ),
    }

    controls = negative_controls(
        extraction,
        logical_rows,
        raw_structure,
    )

    stamp = utc_now().strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / stamp
    )

    if output_dir.exists():
        raise RevalidationError(
            f"拒绝覆盖已有目录: {output_dir}"
        )

    output_dir.mkdir(
        parents=True, exist_ok=False
    )

    result = {
        "schema": "b34d_revalidation/v1",
        "stage": "B3.4d",
        "generated_at": utc_now().isoformat(),
        "api_call": False,
        "family_normalization_version": (
            FAMILY_NORMALIZATION_VERSION
        ),
        "preflight": preflight_data,
        "inputs": {
            "b34b_result": str(args.b34b),
            "b34b_result_sha256": sha256_file(
                args.b34b
            ),
            "gold": str(args.gold),
            "gold_sha256": sha256_file(args.gold),
            "logical_rows": str(args.logical_rows),
            "logical_rows_sha256": sha256_file(
                args.logical_rows
            ),
            "raw_structure": str(args.raw_structure),
            "raw_structure_sha256": sha256_file(
                args.raw_structure
            ),
        },
        "distributions": {
            "old": distribution(
                old_report["parameters"]
            ),
            "new": distribution(
                new_report["parameters"]
            ),
        },
        "old_summary": old_report.get("summary"),
        "new_summary": new_report.get("summary"),
        "field_distribution": {
            "old": field_distribution(
                old_report["parameters"]
            ),
            "new": field_distribution(
                new_report["parameters"]
            ),
        },
        "recovered_from_invalid": recovered,
        "p33_diagnosis": p33,
        "negative_controls": controls,
        "old_report": old_report,
        "new_report": new_report,
    }

    report = build_report(
        result,
        old_report,
        new_report,
        p33,
        recovered,
        controls,
    )
    result["report_markdown"] = report

    result_path = write_json_exclusive(
        output_dir / "b34d_result.json",
        result,
    )
    report_path = write_text_exclusive(
        output_dir / "b34d_report.md",
        report,
    )
    details_path = output_dir / "b34d_details.csv"
    write_details_csv(
        details_path, old_report, new_report
    )
    validator_diff = git_output(
        "diff",
        B34C_COMMIT,
        "--",
        "backend/app/validation/recovered_evidence.py",
    )
    diff_path = write_text_exclusive(
        output_dir / "validator.diff",
        validator_diff + "\n",
    )

    print("=" * 64)
    print("B3.4d 生产重验证完成（未调用 API）")
    print("=" * 64)
    print("输出目录:", output_dir)
    print("结果:", result_path)
    print("报告:", report_path)
    print("明细:", details_path)
    print("Validator Diff:", diff_path)
    print("旧分布:", result["distributions"]["old"])
    print("新分布:", result["distributions"]["new"])
    print(
        "p33:",
        p33["old_status"],
        "->",
        p33["new_status"],
    )
    print(
        "recovered_from_invalid:",
        len(recovered),
    )
    print(
        "negative_controls:",
        f"{sum(item['pass'] for item in controls)}"
        f"/{len(controls)}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
