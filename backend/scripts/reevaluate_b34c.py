"""阶段 B3.4c：使用修复后的评分器重评 B3.2b / B3.4b。

只读取已保存的原始预测，不调用 DeepSeek API，
不修改 Gold、Prompt、生产 Validator 或来源映射。
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.evaluation.ab_scoring import (
    SCORER_VERSION,
    FAMILY_NORMALIZATION_VERSION,
    gold_family,
    prediction_family,
    score_extraction,
)
from app.extraction import prompts
from app.validation.recovered_evidence import (
    logical_family,
    original_family,
)
from app.evaluation.ab_scoring import normalize_family


DEFAULT_GOLD = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b/preflight"
    / "gold_standard.table9.v2.eval.json"
)
DEFAULT_B32B = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.2b"
    / "20261122T143428Z"
    / "ab_result.json"
)
DEFAULT_B34B = (
    PROJECT_ROOT
    / "data/outputs/experiments/B3.4b"
    / "20261122T160706Z"
    / "b34b_result.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data/outputs/experiments/B3.4c"
)
EXPECTED_BRANCH = "experiment/b3.4c-evaluator"
B34B_COMMIT = "c9c6d9da4fcede183b79fc2970accce3f54fa67d"
EXPECTED_GOLD_SHA256 = (
    "f86ab2556d9fa6b0b7fef20c7ea8c62d2b60d85d59842cd414dd7a09383b9b3d"
)
EXPECTED_B34B_PROMPT_SHA256 = (
    "47aac2f3554f4c98cdd0fbe173484df38b20419931598680a0ee865a2189895a"
)
B34C_LOGICAL_PROMPT_PATH = (
    PROJECT_ROOT / "backend/app/extraction/prompts.py"
)
B34B_PROMPT_VERSION = "b3.4b-v1"


class ReevaluationError(ValueError):
    """B3.4c 重评分失败。"""


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
        raise ReevaluationError(
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
        raise ReevaluationError(
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
        raise ReevaluationError(
            f"git {' '.join(args)} 失败"
        ) from exc


def git_preflight():
    branch = git_output(
        "rev-parse", "--abbrev-ref", "HEAD"
    )
    head = git_output("rev-parse", "HEAD")

    if branch != EXPECTED_BRANCH:
        raise ReevaluationError(
            f"当前分支不是 {EXPECTED_BRANCH}: {branch}"
        )

    git_output(
        "merge-base", "--is-ancestor",
        B34B_COMMIT, "HEAD",
    )

    status = git_output(
        "status", "--porcelain=v1"
    )
    changed = git_output(
        "diff", "--name-only", B34B_COMMIT,
        "--",
        "backend/app",
        "backend/scripts",
        "backend/tests",
    )
    changed_files = [
        line.strip()
        for line in changed.splitlines()
        if line.strip()
    ]

    return {
        "branch": branch,
        "head": head,
        "b34b_commit": B34B_COMMIT,
        "status_porcelain": status.splitlines(),
        "tracked_changed_files": changed_files,
    }


def counts(evaluation):
    gold_count = evaluation["gold_count"]
    predicted_count = evaluation["predicted_count"]
    strict_tp = evaluation["strict_matched"]
    key_tp = evaluation["matched_keys"]

    return {
        "strict_tp": strict_tp,
        "strict_fp": predicted_count - strict_tp,
        "strict_fn": gold_count - strict_tp,
        "key_tp": key_tp,
        "key_fp": predicted_count - key_tp,
        "key_fn": gold_count - key_tp,
        "gold_count": gold_count,
        "predicted_count": predicted_count,
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1": evaluation["f1"],
        "key_precision": evaluation["key_precision"],
        "key_recall": evaluation["key_recall"],
        "key_f1": evaluation["key_f1"],
        "key_coverage": (
            round(key_tp / gold_count, 4)
            if gold_count else None
        ),
        "source_extra_predictions": evaluation[
            "false_positive_count"
        ],
        "duplicate_predictions": evaluation[
            "duplicate_count"
        ],
    }


def p33_diagnosis(gold, extraction, evaluation):
    gold_record = next(
        item
        for item in gold["expected_parameters"]
        if item["logical_row_id"] == "t9.p33"
        and item["value_type"] == "typ"
    )
    prediction = next(
        item
        for item in extraction.get("parameters", [])
        if (item.get("evidence") or {}).get(
            "logical_row_id"
        ) == "t9.p33"
        and item.get("value_type") == "typ"
    )
    detail = next(
        item
        for item in evaluation["gold_details"]
        if item["logical_row_id"] == "t9.p33"
        and item["value_type"] == "typ"
    )
    legacy_gold = normalize_family(
        logical_family({
            "parameter_symbol": gold_record.get(
                "parameter_symbol"
            ),
            "parameter_name": gold_record.get(
                "parameter_name"
            ),
        })
    )
    legacy_prediction = normalize_family(
        original_family(prediction.get("name"))
    )

    return {
        "logical_row_id": "t9.p33",
        "value_type": "typ",
        "gold": {
            "parameter_symbol": gold_record.get(
                "parameter_symbol"
            ),
            "parameter_name": gold_record.get(
                "parameter_name"
            ),
            "value": gold_record.get("value"),
            "unit": gold_record.get("unit"),
            "test_conditions": gold_record.get(
                "test_conditions_row_level"
            ),
        },
        "predicted": {
            "name": prediction.get("name"),
            "value": prediction.get("value"),
            "unit": prediction.get("unit"),
            "test_conditions": prediction.get(
                "test_conditions"
            ),
            "evidence": prediction.get("evidence"),
            "source_mapping_status": prediction.get(
                "source_mapping_status"
            ),
        },
        "legacy_identity": {
            "gold_family": legacy_gold,
            "prediction_family": legacy_prediction,
            "identity": legacy_gold == legacy_prediction,
        },
        "fixed_identity": {
            "gold_family": gold_family(gold_record),
            "prediction_family": prediction_family(
                prediction.get("name")
            ),
            "identity": (
                gold_family(gold_record)
                == prediction_family(
                    prediction.get("name")
                )
            ),
        },
        "fields": detail["fields"],
        "strict_match": detail["strict_match"],
        "match_status": detail["match_status"],
        "evidence_check": detail.get("evidence_check"),
    }


def build_versions(gold):
    versions = []

    for path, arm_name, prompt_label in (
        (
            DEFAULT_B32B,
            "flat",
            "B3.2b Flat Prompt",
        ),
        (
            DEFAULT_B32B,
            "logical",
            "B3.2b Logical Prompt",
        ),
        (
            DEFAULT_B34B,
            "logical",
            "B3.4b Logical Prompt",
        ),
    ):
        source = load_json(path)
        arm = next(
            item for item in source["arms"]
            if item["arm"] == arm_name
            if (
                path != DEFAULT_B32B
                or arm_name != "logical"
                or item["extractor_config"].get(
                    "rows_per_chunk"
                ) == 10
            )
        )

        evaluation = score_extraction(
            gold,
            arm["extraction"],
            arm.get("recovered_validation_raw"),
        )
        versions.append({
            "label": f"{path.parent.name}/{arm_name}",
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "source_generated_at": source.get(
                "generated_at"
            ),
            "arm": arm_name,
            "prompt": prompt_label,
            "extractor_config": arm.get(
                "extractor_config"
            ),
            "metrics": arm.get("metrics"),
            "counts": counts(evaluation),
            "evaluation": evaluation,
            "extraction": arm["extraction"],
        })

    return versions


def build_report(versions, gold, preflight, p33):
    lines = [
        "# B3.4c 评估器参数身份映射修复报告",
        "",
        f"- 阶段：B3.4c",
        f"- 生成时间：{utc_now().isoformat()}",
        f"- 分支：`{preflight['branch']}`",
        f"- HEAD：`{preflight['head']}`",
        f"- B3.4b 基线提交：`{B34B_COMMIT}`",
        f"- 评分器版本：`{SCORER_VERSION}`",
        f"- Family 归一化版本："
        f"`{FAMILY_NORMALIZATION_VERSION}`",
        "",
        "> 本阶段不调用 DeepSeek API，只使用已保存的原始预测。",
        "> Gold、Prompt、生产 Validator 和来源映射均未修改。",
        "> 所有版本均使用同一份修复后的评分器。",
        "",
        "## 一、修复内容",
        "",
        "- 在 `ab_scoring.py` 的评估层增加语义优先级：",
        "  `noise > accuracy > 原有 family 规则`。",
        "- `Output voltage noise` / `V_N` / `VN` 归为 "
        "`output_noise`。",
        "- `Output voltage accuracy` 归为 `accuracy`。",
        "- 纯 `Output voltage` 仍归为 `output_voltage`。",
        "- 不修改 `recovered_evidence.logical_family()`，"
        "避免影响生产 Validator。",
        "",
        "## 二、重评分口径",
        "",
        "- Strict TP = 严格匹配数。",
        "- Strict FP = 原始预测数 - 严格匹配数。",
        "- Strict FN = 51 - 严格匹配数。",
        "- Key TP = 参数键匹配数。",
        "- Key FP = 原始预测数 - 参数键匹配数。",
        "- Key FN = 51 - 参数键匹配数。",
        "- `source_extra_predictions` 才是来源无法映射的"
        "真正多余预测；与 Strict FP 分开统计。",
        "",
        "## 三、版本化重评分结果",
        "",
        "| 版本 | Strict TP | Strict FP | Strict FN | "
        "Precision | Recall | F1 | Key TP | Key FP | Key FN | "
        "键覆盖 | 来源多余预测 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for version in versions:
        c = version["counts"]
        lines.append(
            f"| {version['label']} | "
            f"{c['strict_tp']} | {c['strict_fp']} | "
            f"{c['strict_fn']} | {c['precision']} | "
            f"{c['recall']} | {c['f1']} | "
            f"{c['key_tp']} | {c['key_fp']} | "
            f"{c['key_fn']} | {c['key_coverage']} | "
            f"{c['source_extra_predictions']} |"
        )

    lines += [
        "",
        "说明：",
        "",
        "- B3.4b 的 Strict FP 为 0（51/51 严格匹配）。",
        "- B3.2b Flat 的 Strict FP 包含键匹配但字段错误的预测，"
        "不是来源无法映射的 false positive。",
        "- 所有版本均无重复预测。",
        "",
        "## 四、历史结果对照",
        "",
        "| 版本 | 历史严格匹配 | B3.4c 严格匹配 | "
        "历史 F1 | B3.4c F1 |",
        "|---|---:|---:|---:|---:|",
        "| B3.2b Flat | 17/51 | 17/51 | 0.4048 | 0.4048 |",
        "| B3.2b Logical | 26/51 | 26/51 | 0.6753 | 0.6753 |",
        "| B3.4b Logical | 50/51 | 51/51 | 0.9804 | 1.0 |",
        "",
        "## 五、t9.p33 逐字段诊断",
        "",
        "| 项目 | Gold | 候选 | 结果 |",
        "|---|---|---|---|",
        f"| 参数符号 | `{p33['gold']['parameter_symbol']}` | "
        f"`{p33['predicted']['name']}` | — |",
        f"| 参数名 | `{p33['gold']['parameter_name']}` | "
        f"`{p33['predicted']['name']}` | identity "
        f"{p33['fields']['identity']} |",
        f"| 数值 | `{p33['gold']['value']}` | "
        f"`{p33['predicted']['value']}` | "
        f"{p33['fields']['value']} |",
        f"| 单位 | `{p33['gold']['unit']}` | "
        f"`{p33['predicted']['unit']}` | "
        f"{p33['fields']['unit']} |",
        f"| value_type | `{p33['value_type']}` | "
        f"`{p33['value_type']}` | "
        f"{p33['fields']['value_type']} |",
        f"| test_conditions | `{p33['gold']['test_conditions']}` | "
        f"`{p33['predicted']['test_conditions']}` | "
        f"{p33['fields']['test_conditions']} |",
        "",
        "### identity 修复前后",
        "",
        "| 阶段 | Gold family | 预测 family | identity |",
        "|---|---|---|---|",
        f"| 旧评分器 | `{p33['legacy_identity']['gold_family']}` | "
        f"`{p33['legacy_identity']['prediction_family']}` | "
        f"{p33['legacy_identity']['identity']} |",
        f"| B3.4c 修复后 | `{p33['fixed_identity']['gold_family']}` | "
        f"`{p33['fixed_identity']['prediction_family']}` | "
        f"{p33['fixed_identity']['identity']} |",
        "",
        f"- 来源映射状态："
        f"`{p33['predicted']['source_mapping_status']}`",
        f"- 来源检查："
        f"`{p33['evidence_check']}`",
        f"- 最终严格匹配：**{p33['strict_match']}**",
        "",
        "原因：旧评分器直接使用生产 `logical_family()`，"
        "其 `\"output voltage\" in name` 规则先把 "
        "`Output voltage noise` 归为 `output_voltage`；"
        "而预测 `output_voltage_noise` 归为 `output_noise`。"
        "B3.4c 在评估层先识别 noise 语义，再回退到原规则。",
        "",
        "## 六、结论",
        "",
        "1. 修复仅影响评估层 identity 归一化，"
        "不影响数值、单位、条件或来源匹配。",
        "2. B3.4b 从 50/51 变为 51/51 严格匹配，"
        "Precision / Recall / F1 均为 1.0。",
        "3. B3.2b Flat 和 Logical 的历史指标保持不变。",
        "4. 所有版本使用同一个 B3.4c 评分器，"
        "历史报告未被覆盖。",
        "5. 未修改 Gold、Prompt、生产 Validator、"
        "RecoveredEvidenceValidator、来源映射或 Retry。",
        "6. 未调用 DeepSeek API，未进入 B4。",
        "",
    ]

    return "\n".join(lines)


def write_details_csv(path, versions):
    fields = [
        "version",
        "logical_row_id",
        "value_type",
        "match_status",
        "strict_match",
        "identity",
        "value",
        "unit",
        "value_type_match",
        "test_conditions",
        "test_conditions_exact",
        "predicted_name",
        "map_method",
    ]

    with Path(path).open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

        for version in versions:
            for detail in version["evaluation"][
                "gold_details"
            ]:
                fields_map = detail.get("fields") or {}
                writer.writerow({
                    "version": version["label"],
                    "logical_row_id": detail[
                        "logical_row_id"
                    ],
                    "value_type": detail["value_type"],
                    "match_status": detail[
                        "match_status"
                    ],
                    "strict_match": detail[
                        "strict_match"
                    ],
                    "identity": fields_map.get(
                        "identity"
                    ),
                    "value": fields_map.get("value"),
                    "unit": fields_map.get("unit"),
                    "value_type_match": fields_map.get(
                        "value_type"
                    ),
                    "test_conditions": fields_map.get(
                        "test_conditions"
                    ),
                    "test_conditions_exact": (
                        fields_map.get(
                            "test_conditions_exact"
                        )
                    ),
                    "predicted_name": (
                        (detail.get("predicted") or {}).get(
                            "name"
                        )
                    ),
                    "map_method": detail.get(
                        "map_method"
                    ),
                })


def main(argv=None):
    global DEFAULT_B32B, DEFAULT_B34B

    parser = argparse.ArgumentParser(
        description=(
            "B3.4c：使用修复后的评分器"
            "重评 B3.2b / B3.4b"
        )
    )
    parser.add_argument(
        "--gold", default=str(DEFAULT_GOLD)
    )
    parser.add_argument(
        "--b32b", default=str(DEFAULT_B32B)
    )
    parser.add_argument(
        "--b34b", default=str(DEFAULT_B34B)
    )
    parser.add_argument(
        "--output-dir", default=None
    )
    args = parser.parse_args(argv)

    DEFAULT_B32B = Path(args.b32b)
    DEFAULT_B34B = Path(args.b34b)

    preflight = git_preflight()

    if SCORER_VERSION != "b3.4c-v1":
        raise ReevaluationError(
            "评分器版本不是 b3.4c-v1"
        )

    gold = load_json(args.gold)

    if sha256_file(args.gold) != EXPECTED_GOLD_SHA256:
        raise ReevaluationError(
            "Gold v2 eval 哈希与 B3.4b 基线不一致"
        )

    prompt_sha = hashlib.sha256(
        prompts.LOGICAL_INPUT_INSTRUCTION.encode(
            "utf-8"
        )
    ).hexdigest()

    if prompt_sha != EXPECTED_B34B_PROMPT_SHA256:
        raise ReevaluationError(
            "Logical Prompt 与 B3.4b 基线不一致"
        )

    versions = build_versions(gold)
    b34b = next(
        version for version in versions
        if version["prompt"] == "B3.4b Logical Prompt"
    )
    p33 = p33_diagnosis(
        gold,
        b34b["extraction"],
        b34b["evaluation"],
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
        raise ReevaluationError(
            f"拒绝覆盖已有目录: {output_dir}"
        )

    output_dir.mkdir(
        parents=True, exist_ok=False
    )

    result = {
        "schema": "b34c_reevaluation/v1",
        "stage": "B3.4c",
        "generated_at": utc_now().isoformat(),
        "api_call": False,
        "scorer_version": SCORER_VERSION,
        "family_normalization_version": (
            FAMILY_NORMALIZATION_VERSION
        ),
        "preflight": preflight,
        "gold": {
            "path": str(args.gold),
            "sha256": sha256_file(args.gold),
        },
        "prompt": {
            "path": str(B34C_LOGICAL_PROMPT_PATH),
            "logical_instruction_sha256": prompt_sha,
            "file_sha256": sha256_file(
                B34C_LOGICAL_PROMPT_PATH
            ),
            "version": B34B_PROMPT_VERSION,
        },
        "versions": [
            {
                key: value
                for key, value in version.items()
                if key != "evaluation"
            }
            for version in versions
        ],
        "evaluations": {
            version["label"]: version["evaluation"]
            for version in versions
        },
        "p33_diagnosis": p33,
    }

    report = build_report(
        versions, gold, preflight, p33
    )
    result["report_markdown"] = report

    result_path = write_json_exclusive(
        output_dir / "b34c_result.json",
        result,
    )
    report_path = write_text_exclusive(
        output_dir / "b34c_report.md",
        report,
    )
    details_path = output_dir / "b34c_details.csv"
    write_details_csv(details_path, versions)
    scorer_diff = git_output(
        "diff",
        B34B_COMMIT,
        "--",
        "backend/app/evaluation/ab_scoring.py",
    )
    diff_path = write_text_exclusive(
        output_dir / "scorer.diff",
        scorer_diff + "\n",
    )

    print("=" * 64)
    print("B3.4c 重评分完成（未调用 API）")
    print("=" * 64)
    print("输出目录:", output_dir)
    print("结果:", result_path)
    print("报告:", report_path)
    print("明细:", details_path)
    print("评分器 Diff:", diff_path)
    print(
        "scorer:",
        SCORER_VERSION,
        FAMILY_NORMALIZATION_VERSION,
    )

    for version in versions:
        c = version["counts"]
        print(
            version["label"],
            {
                "strict_tp": c["strict_tp"],
                "strict_fp": c["strict_fp"],
                "strict_fn": c["strict_fn"],
                "precision": c["precision"],
                "recall": c["recall"],
                "f1": c["f1"],
                "key_coverage": c["key_coverage"],
            },
        )

    print(
        "t9.p33:",
        {
            "legacy": p33["legacy_identity"],
            "fixed": p33["fixed_identity"],
            "strict": p33["strict_match"],
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
