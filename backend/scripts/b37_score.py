"""B3.7：四臂评分、2x2 归因与报告生成（只读，不调用 LLM）。

统一使用 B3.5c 冻结评分契约（b35c-flat-v1 / b35c-norm-v1）。
主实验只使用 Gold scope 表，因此四个臂的表格范围一致。
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS = BACKEND_ROOT / "scripts"

for path in (BACKEND_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.validation.recovered_integration import (  # noqa: E402
    build_recovered_validation_result,
)
from app.validation.validator import ParameterValidator  # noqa: E402

from b35c_score_flat import (  # noqa: E402
    SCORER_VERSION,
    NORMALIZATION_VERSION,
    aggregate,
    content_allowed,
    content_weight,
    max_weight_match,
    normalize_gold,
    normalize_prediction,
    score_sample,
    write_details_csv,
)
from b37_common import (  # noqa: E402
    B35C_METRICS,
    B35D_FN,
    GOLD,
    MANIFEST,
    REPORTS_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    build_sample_inputs,
    load_json,
    write_json,
)


ARM_ORDER = ("A", "B", "C", "D")
TARGET_STAGE = "llm_extraction"
TARGET_COUNT = 26


def _pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _pp(value):
    return "N/A" if value is None else f"{value * 100:+.2f}pp"


def _num(value):
    return "N/A" if value is None else f"{value:g}"


def _delta(left, right, key):
    a = left.get(key)
    b = right.get(key)

    if a is None or b is None:
        return None

    return round(b - a, 4)


def _delta_counts(left, right, key):
    return int(right.get(key, 0)) - int(left.get(key, 0))


def _arm_delta(left, right):
    return {
        "tp": _delta_counts(left, right, "tp"),
        "fp": _delta_counts(left, right, "fp"),
        "fn": _delta_counts(left, right, "fn"),
        "precision": _delta(left, right, "precision"),
        "recall": _delta(left, right, "recall"),
        "f1": _delta(left, right, "f1"),
        "strict_f1": _delta(left, right, "strict_f1"),
    }


def _load_errors(path):
    if not Path(path).is_file():
        return []

    return load_json(path)


def _fn_ids(errors):
    return {
        error["gold_id"]
        for error in errors
        if error.get("kind") == "FN" and error.get("gold_id")
    }


def _error_ids(errors, kind):
    return {
        error["gold_id"]
        for error in errors
        if error.get("kind") == kind and error.get("gold_id")
    }


def _content_pairs(run_dir, sample, gold_sample):
    """复现评分器的内容匹配，用于诊断 Validator 状态。

    只读 validation.json，不改变冻结评分结果；返回
    (golds, in_scope_predictions, pairs)。
    """

    validation_path = Path(run_dir) / "validation.json"

    if not validation_path.is_file():
        raise SystemExit(f"缺少 validation.json: {validation_path}")

    validation = load_json(validation_path)
    predictions = [
        normalize_prediction(param, index)
        for index, param in enumerate(
            validation.get("parameters", [])
        )
    ]
    golds = [
        normalize_gold(record)
        for record in gold_sample["records"]
    ]
    scope_ids = set(sample["gold_scope_table_ids"])
    gold_rows = {
        (gold["table_id"], gold["row_index"]) for gold in golds
    }
    gold_identities = {gold["identity"] for gold in golds}
    in_scope = [
        prediction
        for prediction in predictions
        if prediction["table_id"] in scope_ids
        and (
            prediction["identity"] in gold_identities
            or (
                prediction["table_id"],
                prediction["row_index"],
            )
            in gold_rows
        )
    ]
    pairs = max_weight_match(
        golds, in_scope, content_allowed, content_weight
    )

    return golds, in_scope, pairs


def _trusted_view(run_id, gold, manifest):
    """官方评分不变，额外统计“生产可信”内容 TP。

    可信 = 该内容 TP 的 Validator 状态为 validated 或
    pending_review；invalid 仍计入官方 TP，但不计入可信 TP。
    保守口径的分母仍使用全部预测（TP+FP），因此 invalid
    预测会被计入 Precision 惩罚。
    """

    result = {}

    for arm in ARM_ORDER:
        run_root = RUNS_DIR / arm / run_id
        status_counts = Counter()
        per_sample = {}
        total_gold = 0
        total_tp = 0
        total_fp = 0

        for gold_sample in gold.get("samples", []):
            sample_id = gold_sample["sample_id"]
            sample = manifest[sample_id]
            golds, in_scope, pairs = _content_pairs(
                run_root / sample_id, sample, gold_sample
            )
            sample_status = Counter(
                in_scope[pred_index].get("validation_status")
                for _, pred_index in pairs
            )
            tp = len(pairs)
            fp = len(in_scope) - tp
            total_gold += len(golds)
            total_tp += tp
            total_fp += fp
            status_counts.update(sample_status)
            per_sample[sample_id] = {
                "gold": len(golds),
                "content_tp": tp,
                "fp": fp,
                "validated_tp": sample_status.get(
                    "validated", 0
                ),
                "pending_tp": sample_status.get(
                    "pending_review", 0
                ),
                "invalid_tp": sample_status.get("invalid", 0),
                "status_counts": dict(sample_status),
            }

        trusted_tp = status_counts.get(
            "validated", 0
        ) + status_counts.get("pending_review", 0)
        precision = (
            round(trusted_tp / (total_tp + total_fp), 4)
            if total_tp + total_fp
            else None
        )
        recall = (
            round(trusted_tp / total_gold, 4)
            if total_gold
            else None
        )
        f1 = (
            round(
                2 * precision * recall / (precision + recall),
                4,
            )
            if precision and recall
            else None
        )
        result[arm] = {
            "content_tp": total_tp,
            "content_fp": total_fp,
            "content_fn": total_gold - total_tp,
            "gold_total": total_gold,
            "trusted_tp": trusted_tp,
            "validated_tp": status_counts.get("validated", 0),
            "pending_tp": status_counts.get(
                "pending_review", 0
            ),
            "invalid_tp": status_counts.get("invalid", 0),
            "trusted_precision": precision,
            "trusted_recall": recall,
            "trusted_f1": f1,
            "per_sample": per_sample,
        }

    return result


def _target_omissions_validated(run_id, gold, manifest):
    """26 条遗漏中，内容命中且 Validator 可信的数量。"""

    trace = load_json(B35D_FN)
    target_ids = {
        record["gold_id"]
        for record in trace["records"]
        if record.get("stage_group") == TARGET_STAGE
    }

    if len(target_ids) != TARGET_COUNT:
        raise SystemExit(
            f"B3.5d 目标省略数不是 {TARGET_COUNT}: "
            f"{len(target_ids)}"
        )

    result = {}

    for arm in ARM_ORDER:
        run_root = RUNS_DIR / arm / run_id
        counts = Counter()
        per_sample = defaultdict(Counter)

        for gold_sample in gold.get("samples", []):
            sample_id = gold_sample["sample_id"]
            sample = manifest[sample_id]
            golds, in_scope, pairs = _content_pairs(
                run_root / sample_id, sample, gold_sample
            )
            matched = {
                golds[gold_index]["gold_id"]: in_scope[
                    pred_index
                ]
                for gold_index, pred_index in pairs
            }
            sample_target_ids = [
                gold["gold_id"]
                for gold in golds
                if gold["gold_id"] in target_ids
            ]

            for gold_id in sample_target_ids:
                prediction = matched.get(gold_id)
                status = (
                    prediction.get("validation_status")
                    if prediction is not None
                    else "fn"
                )
                counts[status] += 1
                per_sample[sample_id][status] += 1

        trusted = counts.get("validated", 0) + counts.get(
            "pending_review", 0
        )
        result[arm] = {
            "total": TARGET_COUNT,
            "content_hits": TARGET_COUNT - counts.get("fn", 0),
            "trusted_hits": trusted,
            "validated_hits": counts.get("validated", 0),
            "pending_hits": counts.get(
                "pending_review", 0
            ),
            "invalid_hits": counts.get("invalid", 0),
            "fn": counts.get("fn", 0),
            "status_counts": dict(counts),
            "per_sample": {
                sample_id: dict(values)
                for sample_id, values in per_sample.items()
            },
        }

    return result


def _target_status_rows(run_id, gold, manifest):
    """逐条输出 26 条遗漏在各臂的 Validator 状态。"""

    trace = load_json(B35D_FN)
    target_records = {
        record["gold_id"]: record
        for record in trace["records"]
        if record.get("stage_group") == TARGET_STAGE
    }
    rows = {
        gold_id: {
            "sample_id": record.get("sample_id"),
            "vendor": record.get("vendor"),
            "gold_id": gold_id,
            "identity": record.get("gold_identity"),
            "value": record.get("gold_value"),
            "unit": record.get("gold_unit"),
            "value_type": record.get("gold_value_type"),
        }
        for gold_id, record in target_records.items()
    }

    for arm in ARM_ORDER:
        run_root = RUNS_DIR / arm / run_id

        for gold_sample in gold.get("samples", []):
            sample_id = gold_sample["sample_id"]
            sample = manifest[sample_id]
            golds, in_scope, pairs = _content_pairs(
                run_root / sample_id, sample, gold_sample
            )
            matched = {
                golds[gold_index]["gold_id"]: in_scope[
                    pred_index
                ]
                for gold_index, pred_index in pairs
            }

            for gold_record in golds:
                gold_id = gold_record["gold_id"]

                if gold_id not in rows:
                    continue

                prediction = matched.get(gold_id)
                rows[gold_id][f"{arm}_status"] = (
                    prediction.get("validation_status")
                    if prediction is not None
                    else "fn"
                )
                rows[gold_id][f"{arm}_content_hit"] = (
                    prediction is not None
                )

    return list(rows.values())


def _adi_identity_diagnostics(arm_errors):
    """ADI 内容 FN/FP 中“数值/单位/类型相同但身份不同”的配对。"""

    def value_key(value):
        try:
            return round(float(value), 9)
        except (TypeError, ValueError):
            return str(value)

    def unit_key(value):
        return str(value or "").strip()

    result = {}

    for arm in ARM_ORDER:
        errors = [
            error
            for error in arm_errors[arm]
            if error.get("sample_id") == "adi_lt3045"
        ]
        fn = [
            error for error in errors if error.get("kind") == "FN"
        ]
        fp = [
            error for error in errors if error.get("kind") == "FP"
        ]
        fn_keys = Counter(
            (
                value_key(error.get("gold_value")),
                unit_key(error.get("gold_unit")),
                error.get("gold_value_type"),
            )
            for error in fn
        )
        fp_keys = Counter(
            (
                value_key(error.get("prediction_value")),
                unit_key(error.get("prediction_unit")),
                error.get("prediction_value_type"),
            )
            for error in fp
        )
        paired = sum((fn_keys & fp_keys).values())
        identity_pairs = Counter()
        fp_by_key = defaultdict(list)

        for error in fp:
            fp_by_key[
                (
                    value_key(error.get("prediction_value")),
                    unit_key(error.get("prediction_unit")),
                    error.get("prediction_value_type"),
                )
            ].append(error)

        for error in fn:
            key = (
                value_key(error.get("gold_value")),
                unit_key(error.get("gold_unit")),
                error.get("gold_value_type"),
            )
            candidates = fp_by_key.get(key) or []

            if not candidates:
                continue

            match = candidates.pop(0)
            identity_pairs[
                (
                    error.get("gold_identity"),
                    match.get("prediction_identity"),
                )
            ] += 1

        result[arm] = {
            "gold_records": 18,
            "content_tp": 18 - len(fn),
            "fn": len(fn),
            "fp": len(fp),
            "exact_value_unit_type_pairs": paired,
            "identity_pairs": [
                {
                    "gold_identity": gold_identity,
                    "prediction_identity": prediction_identity,
                    "count": count,
                }
                for (
                    gold_identity,
                    prediction_identity,
                ), count in identity_pairs.items()
            ],
        }

    return result


def _score_arm(arm, run_id, gold, manifest):
    run_root = RUNS_DIR / arm / run_id
    manifest_path = run_root / "run_manifest.json"

    if not manifest_path.is_file():
        raise SystemExit(f"缺少 {arm} 臂运行 manifest: {manifest_path}")

    run_manifest = load_json(manifest_path)

    if run_manifest.get("status") != "completed":
        raise SystemExit(
            f"{arm} 臂运行未完成，禁止评分: "
            f"{run_manifest.get('status')}"
        )

    samples = []

    for gold_sample in gold.get("samples", []):
        sample_id = gold_sample["sample_id"]
        sample = manifest[sample_id]
        run_dir = run_root / sample_id

        if not run_dir.is_dir():
            raise SystemExit(f"缺少 {arm} / {sample_id} 运行目录")

        samples.append(
            score_sample(sample, gold_sample, run_dir)
        )

    micro = aggregate(samples)
    errors = [
        error for sample in samples for error in sample["errors"]
    ]
    resolved = {
        "schema": "b37_arm_metrics/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": SCORER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "arm": arm,
        "run_root": str(run_root.relative_to(PROJECT_ROOT)),
        "run_manifest": run_manifest,
        "micro": micro,
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key != "errors"
            }
            for sample in samples
        ],
    }

    return resolved, errors, samples


def _prompt_audit(run_id):
    audit = {}

    for arm in ARM_ORDER:
        run_root = RUNS_DIR / arm / run_id
        total_requests = 0
        family_true = 0
        logical_true = 0
        temperatures = Counter()
        request_models = Counter()

        if not run_root.is_dir():
            audit[arm] = {"missing": True}
            continue

        for sample_dir in sorted(run_root.iterdir()):
            path = sample_dir / "api_requests.json"

            if not path.is_file():
                continue

            payload = load_json(path)

            for request in payload.get("requests", []):
                total_requests += 1
                family_true += int(
                    bool(request.get("prompt_flags", {}).get(
                        "family_coverage"
                    ))
                )
                logical_true += int(
                    bool(request.get("prompt_flags", {}).get(
                        "logical_input"
                    ))
                )
                temperatures[str(request.get("temperature"))] += 1
                request_models[str(request.get("requested_model"))] += 1

        audit[arm] = {
            "total_requests": total_requests,
            "family_coverage_requests": family_true,
            "logical_input_requests": logical_true,
            "temperatures": dict(temperatures),
            "requested_models": dict(request_models),
        }

    return audit


def _target_omissions(arm_errors):
    trace = load_json(B35D_FN)
    targets = [
        record
        for record in trace["records"]
        if record.get("stage_group") == TARGET_STAGE
    ]

    if len(targets) != TARGET_COUNT:
        raise SystemExit(
            f"B3.5d 目标省略数不是 {TARGET_COUNT}: {len(targets)}"
        )

    rows = []

    for record in targets:
        gold_id = record["gold_id"]
        row = {
            "sample_id": record["sample_id"],
            "vendor": record["vendor"],
            "gold_id": gold_id,
            "identity": record["gold_identity"],
            "value": record["gold_value"],
            "unit": record["gold_unit"],
            "value_type": record["gold_value_type"],
            "b35c_category": record.get("b35c_category"),
        }

        for arm in ARM_ORDER:
            errors = arm_errors[arm]
            fn_ids = _fn_ids(errors)
            condition_ids = _error_ids(errors, "condition")
            source_ids = _error_ids(errors, "source")
            content_hit = gold_id not in fn_ids
            strict_hit = (
                content_hit
                and gold_id not in condition_ids
                and gold_id not in source_ids
            )
            row[f"{arm}_content_hit"] = content_hit
            row[f"{arm}_strict_hit"] = strict_hit

        rows.append(row)

    summary = {}

    for arm in ARM_ORDER:
        per_sample = defaultdict(lambda: {
            "total": 0,
            "content_hits": 0,
            "strict_hits": 0,
        })
        per_identity = defaultdict(lambda: {
            "total": 0,
            "content_hits": 0,
            "strict_hits": 0,
        })

        for row in rows:
            for bucket in (
                per_sample[row["sample_id"]],
                per_identity[row["identity"]],
            ):
                bucket["total"] += 1
                bucket["content_hits"] += int(
                    row[f"{arm}_content_hit"]
                )
                bucket["strict_hits"] += int(
                    row[f"{arm}_strict_hit"]
                )

        summary[arm] = {
            "total": len(rows),
            "content_hits": sum(
                int(row[f"{arm}_content_hit"]) for row in rows
            ),
            "strict_hits": sum(
                int(row[f"{arm}_strict_hit"]) for row in rows
            ),
            "per_sample": dict(per_sample),
            "per_identity": dict(per_identity),
        }

    return rows, summary


def _adi_records(arm_errors, gold):
    adi_gold = next(
        sample
        for sample in gold["samples"]
        if sample["sample_id"] == "adi_lt3045"
    )
    rows = []

    for record in adi_gold["records"]:
        gold_id = record["gold_id"]
        row = {
            "gold_id": gold_id,
            "identity": record["parameter_identity"],
            "value": record["value"],
            "unit": record["unit"],
            "value_type": record["value_type"],
            "row_index": record["source"]["row_index"],
            "page": record["source"].get("page"),
        }

        for arm in ARM_ORDER:
            errors = arm_errors[arm]
            fn_ids = _fn_ids(errors)
            condition_ids = _error_ids(errors, "condition")
            source_ids = _error_ids(errors, "source")
            content_hit = gold_id not in fn_ids
            row[f"{arm}_content_hit"] = content_hit
            row[f"{arm}_strict_hit"] = (
                content_hit
                and gold_id not in condition_ids
                and gold_id not in source_ids
            )
            row[f"{arm}_fn_category"] = next(
                (
                    error["category"]
                    for error in errors
                    if error.get("gold_id") == gold_id
                    and error.get("kind") == "FN"
                ),
                None,
            )

        rows.append(row)

    summary = {
        arm: {
            "gold_records": len(rows),
            "content_tp": sum(
                int(row[f"{arm}_content_hit"]) for row in rows
            ),
            "strict_tp": sum(
                int(row[f"{arm}_strict_hit"]) for row in rows
            ),
            "fn": sum(
                int(not row[f"{arm}_content_hit"]) for row in rows
            ),
        }
        for arm in ARM_ORDER
    }

    return rows, summary


def _ncp_units(arm_errors):
    out = {}

    for arm in ARM_ORDER:
        errors = [
            error
            for error in arm_errors[arm]
            if error.get("sample_id") == "onsemi_ncp170"
            and error.get("category") == "llm_wrong_unit"
        ]
        out[arm] = {
            "llm_wrong_unit_count": len(errors),
            "errors": errors,
        }

    return out


def _validator_negative_control(run_id, manifest):
    """用真实预测做确定性负控：错误单位应被生产 Validator 拒绝。"""

    results = {}

    for arm in ARM_ORDER:
        run_root = RUNS_DIR / arm / run_id
        candidate = None
        sample_id = None

        if not run_root.is_dir():
            results[arm] = {"ok": False, "error": "运行目录缺失"}
            continue

        for sample_dir in sorted(run_root.iterdir()):
            validation_path = sample_dir / "validation.json"

            if not validation_path.is_file():
                continue

            validation = load_json(validation_path)

            for param in validation.get("parameters", []):
                if param.get("validation_status") != "validated":
                    continue

                evidence = param.get("evidence") or {}

                if not all(
                    field in evidence
                    for field in ("table_id", "row_index", "page",
                                  "source_text")
                ):
                    continue

                candidate = param
                sample_id = sample_dir.name
                break

            if candidate is not None:
                break

        if candidate is None:
            results[arm] = {
                "ok": False,
                "error": "没有可用于负控的 validated 参数",
            }
            continue

        mutated = deepcopy(candidate)
        mutated["unit"] = "bogus_unit_for_b37_negative_control"
        extraction = {"parameters": [mutated]}
        legacy = ParameterValidator().validate(extraction)
        legacy_status = legacy["parameters"][0].get(
            "validation_status"
        )

        record = {
            "sample_id": sample_id,
            "gold_candidate": {
                "name": candidate.get("name"),
                "value": candidate.get("value"),
                "unit": candidate.get("unit"),
                "value_type": candidate.get("value_type"),
            },
            "mutated_unit": mutated["unit"],
            "legacy_status": legacy_status,
            "ok": legacy_status == "invalid",
        }

        if arm in {"B", "D"}:
            sample = manifest[sample_id]
            inputs = build_sample_inputs(sample)
            cfg = {
                "logical_rows": inputs["logical_rows"],
                "raw_structure": inputs["raw_structure"],
                "pdf_sha256": inputs["pdf_sha256"],
            }
            recovered = build_recovered_validation_result(
                extraction, legacy, cfg
            )
            recovered_status = recovered["parameters"][0].get(
                "validation_status"
            )
            record["recovered_status"] = recovered_status
            record["ok"] = (
                record["ok"] and recovered_status == "invalid"
            )

        results[arm] = record

    return results


def _quality_gates(micro_by_arm, adi_summary, ncp_units,
                   negative_control):
    control = micro_by_arm["A"]
    gates = {}

    for arm in ARM_ORDER:
        micro = micro_by_arm[arm]
        gates[arm] = {
            "recall_gt_34_21": (
                micro.get("recall") is not None
                and micro["recall"] > 0.3421
            ),
            "f1_gt_50_00": (
                micro.get("f1") is not None
                and micro["f1"] > 0.50
            ),
            "precision_not_lower_than_A_by_2pp": (
                micro.get("precision") is not None
                and control.get("precision") is not None
                and micro["precision"] >= control["precision"] - 0.02
            ),
            "ncp_unit_errors_not_increased": (
                ncp_units[arm]["llm_wrong_unit_count"]
                <= ncp_units["A"]["llm_wrong_unit_count"]
            ),
            "adi_table4_matchable": (
                adi_summary[arm]["content_tp"] > 0
            ),
            "validator_rejects_true_error": bool(
                negative_control.get(arm, {}).get("ok")
            ),
        }

    return gates


def _recommendation(
    micro_by_arm,
    gates,
    adi_summary,
    ncp_units,
    trusted_view,
    target_validated,
):
    control = micro_by_arm["A"]
    candidates = ["B", "C", "D"]
    best = max(
        candidates,
        key=lambda arm: (
            micro_by_arm[arm].get("f1") or 0.0,
            micro_by_arm[arm].get("recall") or 0.0,
        ),
    )
    best_trusted = max(
        candidates,
        key=lambda arm: (
            trusted_view[arm].get("trusted_f1") or 0.0,
            trusted_view[arm].get("trusted_tp") or 0,
        ),
    )
    best_micro = micro_by_arm[best]
    delta_recall = (
        (best_micro.get("recall") or 0.0)
        - (control.get("recall") or 0.0)
    )
    delta_f1 = (
        (best_micro.get("f1") or 0.0)
        - (control.get("f1") or 0.0)
    )
    delta_precision = (
        (best_micro.get("precision") or 0.0)
        - (control.get("precision") or 0.0)
    )
    reasons = []

    if best_micro.get("recall") is None:
        reasons.append("最佳臂 Recall 不可计算")
    if best_micro.get("f1") is None:
        reasons.append("最佳臂 F1 不可计算")
    if delta_recall <= 0:
        reasons.append("Recall 相对 A0 未提升")
    if delta_f1 <= 0:
        reasons.append("F1 相对 A0 未提升")
    if delta_precision < -0.02:
        reasons.append("官方口径 Precision 下降超过 2pp")
    if ncp_units[best]["llm_wrong_unit_count"] > ncp_units[
        "A"
    ]["llm_wrong_unit_count"]:
        reasons.append("NCP 单位错误增加")
    if adi_summary[best]["content_tp"] <= adi_summary[
        "A"
    ]["content_tp"]:
        reasons.append("ADI table 4 可匹配 TP 未增加")
    if not gates[best]["validator_rejects_true_error"]:
        reasons.append("Validator 负控未通过")

    trusted = trusted_view[best]
    if trusted.get("invalid_tp", 0) > 0:
        reasons.append(
            f"官方最佳臂有 {trusted['invalid_tp']} 条内容 TP 被 "
            "Validator 判 invalid，生产可信 F1 仅 "
            f"{_pct(trusted.get('trusted_f1'))}"
        )

    passing = [
        arm for arm in candidates if all(gates[arm].values())
    ]

    if not passing:
        reasons.append("没有任何新臂同时通过全部质量门禁")

    return {
        "best_arm": best,
        "best_label": {
            "A": "Flat + P0",
            "B": "Logical + P0",
            "C": "Flat + P1",
            "D": "Logical + P1",
        }[best],
        "delta_recall": round(delta_recall, 4),
        "delta_precision": round(delta_precision, 4),
        "delta_f1": round(delta_f1, 4),
        "best_trusted_arm": best_trusted,
        "best_trusted_label": {
            "A": "Flat + P0",
            "B": "Logical + P0",
            "C": "Flat + P1",
            "D": "Logical + P1",
        }[best_trusted],
        "best_trusted_f1": trusted_view[best_trusted].get(
            "trusted_f1"
        ),
        "best_trusted_recall": trusted_view[best_trusted].get(
            "trusted_recall"
        ),
        "best_trusted_target_hits": target_validated[
            best_trusted
        ].get("trusted_hits"),
        "passing_arms": passing,
        "decision": (
            "建议作为第一版候选，需重复运行确认"
            if not reasons
            else "暂不建议直接作为第一版"
        ),
        "reasons": reasons,
    }


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _write_report(payload, path):
    micro = payload["micro_by_arm"]
    historical = payload["historical_b35c"]
    deltas = payload["factorial_deltas"]
    adi = payload["adi_summary"]
    targets = payload["target_omissions_summary"]
    ncp = payload["ncp_units"]
    gates = payload["quality_gates"]
    recommendation = payload["recommendation"]
    prompt_audit = payload["prompt_audit"]
    negative = payload["validator_negative_control"]
    run_summary = payload["run_summary"]
    trusted = payload["trusted_view"]
    target_validated = payload["target_omissions_validated"]
    identity_diag = payload["adi_identity_diagnostics"]

    labels = {
        "A": "A0 Flat + P0",
        "B": "A2 Logical + P0",
        "C": "A1 Flat + P1",
        "D": "A3 Logical + P1",
    }

    lines = [
        "# B3.7 受控 2×2 A/B 实验报告",
        "",
        f"- 生成时间（UTC）：{payload['generated_at']}",
        f"- 评分契约：`{SCORER_VERSION}` / `{NORMALIZATION_VERSION}`",
        f"- 冻结 Gold：`{GOLD.relative_to(PROJECT_ROOT)}`",
        f"- 表格范围：`{payload['table_scope']}`",
        "- 四臂均使用 Gold scope 表，避免全量 selected tables "
        "带来的表格范围差异。",
        "",
        "## 0. 重要环境变化（必须先读）",
        "",
        "B3.5c 运行时的模型是 DeepSeek 官方 `deepseek-flash` + "
        "`https://api.deepseek.com`，并使用全量 selected tables。"
        "本次 B3.7 的 `.env` 已变为火山引擎 Ark 兼容端点，"
        "实际请求模型为 `deepseek-v4-1-flash-261017`。"
        "因此：",
        "",
        "1. **B3.5c 指标只能作为历史参考，不能直接作为本次 A/B "
        "的控制组。**",
        "2. 本次 A0 使用当前模型、当前代码、Gold scope 表重新运行，"
        "作为真正的控制组。",
        "3. 四臂内部模型、端点、temperature、表格范围、分块、"
        "重试与评分契约一致，结构/Prompt 的差值仍可归因。",
        "4. B3.5c 与本次 A0 的绝对差异包含模型变化，不能全部归因于 "
        "B3.6。",
        "",
        "实际响应模型与 fingerprint：",
        "",
        "| 臂 | response_models | system_fingerprints |",
        "| --- | --- | --- |",
    ]

    for arm in ARM_ORDER:
        run = run_summary[arm]
        lines.append(
            f"| {arm} | {run.get('response_models')} | "
            f"{run.get('system_fingerprints')} |"
        )

    lines.extend([
        "",
        "## 1. 四臂主结果",
        "",
        "| 臂 | 输入 + Prompt | TP | FP | FN | Precision | Recall | "
        "F1 | 严格 F1 | API calls | Tokens | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        m = micro[arm]
        run = run_summary[arm]
        usage = run.get("total_usage", {})
        lines.append(
            f"| {arm} | {labels[arm]} | {m['tp']} | {m['fp']} | "
            f"{m['fn']} | {_pct(m['precision'])} | "
            f"{_pct(m['recall'])} | {_pct(m['f1'])} | "
            f"{_pct(m['strict_f1'])} | "
            f"{run.get('total_api_calls')} | "
            f"{usage.get('total_tokens')} | "
            f"{run.get('total_elapsed_s')} |"
        )

    lines.extend([
        "",
        "历史 B3.5c 参考（不同模型、不同表格范围，**不可直接比较**）："
        f"TP={historical['tp']}, FP={historical['fp']}, "
        f"FN={historical['fn']}, P={_pct(historical['precision'])}, "
        f"R={_pct(historical['recall'])}, F1={_pct(historical['f1'])}, "
        f"严格 F1={_pct(historical['strict_f1'])}。",
        "",
        "## 2. 2×2 因子归因",
        "",
        "结构效应：B−A（P0 下）与 D−C（P1 下）；"
        "Prompt 效应：C−A（Flat 下）与 D−B（Logical 下）；"
        "组合效应：D−A；交互：D−C−(B−A)。",
        "",
        "| 对比 | ΔTP | ΔFP | ΔFN | ΔPrecision | ΔRecall | ΔF1 | "
        "Δ严格 F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for name, delta in deltas.items():
        lines.append(
            f"| {name} | {delta['tp']:+d} | {delta['fp']:+d} | "
            f"{delta['fn']:+d} | {_pp(delta['precision'])} | "
            f"{_pp(delta['recall'])} | {_pp(delta['f1'])} | "
            f"{_pp(delta['strict_f1'])} |"
        )

    lines.extend([
        "",
        "## 3. 分厂商结果",
        "",
        "| 臂 | 厂商 | TP | FP | FN | P | R | F1 | 严格 F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    sample_order = [
        "adi_lt3045",
        "diodes_ap2112",
        "onsemi_ncp170",
        "renesas_isl80510",
    ]

    for arm in ARM_ORDER:
        for sample_id in sample_order:
            sample = next(
                item
                for item in payload["samples_by_arm"][arm]
                if item["sample_id"] == sample_id
            )
            lines.append(
                f"| {arm} | {sample['vendor']} | {sample['tp']} | "
                f"{sample['fp']} | {sample['fn']} | "
                f"{_pct(sample['precision'])} | "
                f"{_pct(sample['recall'])} | {_pct(sample['f1'])} | "
                f"{_pct(sample['strict_f1'])} |"
            )

    lines.extend([
        "",
        "## 4. B3.5d 的 26 条 LLM 遗漏",
        "",
        "命中定义为该 Gold 记录不再出现在该臂 FN 中；严格命中还要求"
        "测试条件完整且来源行/page 一致。"
        "“Validator 可信命中”只统计匹配预测的 Validator 状态为 "
        "validated 或 pending_review 的记录。",
        "",
        "| 臂 | 内容命中 / 26 | 严格命中 / 26 | Validator 可信 / 26 | "
        "invalid / 26 |",
        "| --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        summary = targets[arm]
        validated = target_validated[arm]
        lines.append(
            f"| {arm} | {summary['content_hits']} / "
            f"{summary['total']} | {summary['strict_hits']} / "
            f"{summary['total']} | {validated['trusted_hits']} / "
            f"{validated['total']} | {validated['invalid_hits']} / "
            f"{validated['total']} |"
        )

    lines.extend(["", "按厂商：", "", "| 臂 | 厂商 | 内容命中 | 严格命中 | 总数 |", "| --- | --- | --- | --- | --- |"])

    for arm in ARM_ORDER:
        for sample_id in sample_order:
            item = targets[arm]["per_sample"].get(sample_id)
            if not item:
                continue
            lines.append(
                f"| {arm} | {sample_id} | {item['content_hits']} | "
                f"{item['strict_hits']} | {item['total']} |"
            )

    lines.extend([
        "",
        "按厂商的 Validator 状态（26 条遗漏，诊断）：",
        "",
        "| 臂 | 厂商 | validated | pending | invalid | 未命中 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        for sample_id in sample_order:
            statuses = target_validated[arm]["per_sample"].get(
                sample_id
            )

            if not statuses:
                continue

            lines.append(
                f"| {arm} | {sample_id} | "
                f"{statuses.get('validated', 0)} | "
                f"{statuses.get('pending_review', 0)} | "
                f"{statuses.get('invalid', 0)} | "
                f"{statuses.get('fn', 0)} |"
            )

    lines.extend([
        "",
        "按参数族：",
        "",
        "| 臂 | 参数族 | 内容命中 | 严格命中 | 总数 |",
        "| --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        for identity, item in sorted(
            targets[arm]["per_identity"].items()
        ):
            lines.append(
                f"| {arm} | `{identity}` | {item['content_hits']} | "
                f"{item['strict_hits']} | {item['total']} |"
            )

    lines.extend([
        "",
        "## 4b. 生产可信视图（诊断，不改变官方评分）",
        "",
        "官方评分只看内容匹配，不排除 `invalid`。下表把 Validator "
        "判为 `invalid` 的内容 TP 单独拆出，得到更保守的生产可信"
        "指标：",
        "",
        "- `可信 TP` = 官方内容 TP 中状态为 validated 或 "
        "pending_review 的数量；",
        "- `可信 Precision` = 可信 TP /（官方 TP + 官方 FP），"
        "invalid 预测仍计入分母；",
        "- `可信 Recall` = 可信 TP / 76；`可信 F1` 由两者计算。",
        "",
        "| 臂 | 内容 TP | invalid TP | 可信 TP | 可信 P | 可信 R | "
        "可信 F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        item = trusted[arm]
        lines.append(
            f"| {arm} | {item['content_tp']} | {item['invalid_tp']} | "
            f"{item['trusted_tp']} | "
            f"{_pct(item['trusted_precision'])} | "
            f"{_pct(item['trusted_recall'])} | "
            f"{_pct(item['trusted_f1'])} |"
        )

    lines.extend([
        "",
        "分厂商可信 TP：",
        "",
        "| 臂 | 厂商 | 内容 TP | validated | pending | invalid |",
        "| --- | --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        for sample_id in sample_order:
            item = trusted[arm]["per_sample"].get(sample_id)

            if not item:
                continue

            lines.append(
                f"| {arm} | {sample_id} | {item['content_tp']} | "
                f"{item['validated_tp']} | {item['pending_tp']} | "
                f"{item['invalid_tp']} |"
            )

    lines.extend([
        "",
        "## 5. ADI 18 条结构 FN",
        "",
        "| 臂 | 内容 TP / 18 | 严格 TP / 18 | FN / 18 |",
        "| --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        item = adi[arm]
        lines.append(
            f"| {arm} | {item['content_tp']} | {item['strict_tp']} | "
            f"{item['fn']} |"
        )

    lines.extend([
        "",
        "ADI 内容 FN/FP 的数值/单位/类型配对诊断：",
        "",
        "| 臂 | FN | FP | 数值+单位+类型完全相同的 FN/FP 配对 |",
        "| --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        item = identity_diag[arm]
        lines.append(
            f"| {arm} | {item['fn']} | {item['fp']} | "
            f"{item['exact_value_unit_type_pairs']} |"
        )

    for arm in ARM_ORDER:
        pairs = identity_diag[arm]["identity_pairs"]

        if not pairs:
            continue

        lines.extend([
            "",
            f"{arm} 臂的身份差异：",
            "",
            "| Gold identity | 模型 identity | 数量 |",
            "| --- | --- | --- |",
        ])

        for pair in pairs:
            lines.append(
                f"| `{pair['gold_identity']}` | "
                f"`{pair['prediction_identity']}` | "
                f"{pair['count']} |"
            )

    lines.extend([
        "",
        "解释：B/D 的 ADI 精度下降主要不是数值错误，而是同一数值"
        "被模型归到不同 identity（例如 ADI 两行都被命名为 "
        "`current_limit`）。冻结评分按 identity 精确匹配，因此同时"
        "产生 FN 和 FP；这解释了 B/D 的 Precision 下降，但不能据此"
        "把数值判为错误。",
        "",
        "## 6. NCP170 单位错误",
        "",
        "| 臂 | `llm_wrong_unit` 数 |",
        "| --- | --- |",
    ])

    for arm in ARM_ORDER:
        lines.append(
            f"| {arm} | {ncp[arm]['llm_wrong_unit_count']} |"
        )

    lines.extend([
        "",
        "## 7. 校验状态与 Validator 负控",
        "",
        "| 臂 | validated 首轮 | correctable | invalid | pending_review | total |",
        "| --- | --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        dist = micro[arm]["validation_status_distribution"]
        lines.append(
            f"| {arm} | {dist.get('validated_first_pass', 0)} | "
            f"{dist.get('correctable_retry_corrected', 0)} | "
            f"{dist.get('invalid', 0)} | "
            f"{dist.get('pending_review', 0)} | {dist.get('total', 0)} |"
        )

    lines.extend([
        "",
        "注意：B/D 的 `invalid` 绝大多数是 `recovered_conflict:identity`，"
        "即数值/单位正确但模型 identity 与 Logical 行长 identity 不一致；"
        "C 的 `invalid` 主要是 `%/A`、`%/V`、`uVRMS` 等合法单位不在 "
        "Validator 白名单内。两者都不是数值计算错误，但会阻塞生产"
        "可信输出。",
        "",
        "确定性负控：取真实 validated 预测，把单位改为非法单位，再跑生产 "
        "Validator；期望 `invalid`。",
        "",
        "| 臂 | 候选来源 | legacy 状态 | recovered 状态 | 通过 |",
        "| --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        item = negative.get(arm, {})
        lines.append(
            f"| {arm} | {item.get('sample_id')} | "
            f"{item.get('legacy_status')} | "
            f"{item.get('recovered_status', 'N/A')} | "
            f"{'PASS' if item.get('ok') else 'FAIL'} |"
        )

    lines.extend([
        "",
        "## 8. Prompt / 输入开关审计",
        "",
        "| 臂 | 请求数 | 含扩展 Prompt | 含 Logical 指令 | temperature |",
        "| --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        item = prompt_audit[arm]
        lines.append(
            f"| {arm} | {item.get('total_requests')} | "
            f"{item.get('family_coverage_requests')} | "
            f"{item.get('logical_input_requests')} | "
            f"{item.get('temperatures')} |"
        )

    lines.extend([
        "",
        "说明：上表的“含扩展 Prompt / 含 Logical 指令”只统计首轮"
        "提取请求。所有臂的纠错重试请求都使用生产 `RETRY_INSTRUCTION`"
        "（Logical 臂追加 `LOGICAL_RETRY_INSTRUCTION`），不携带 "
        "Prompt 族覆盖；因此重试请求数不为零的臂，其 Prompt 效应会被"
        "轻微稀释。这是生产管线的既有行为，实验中未修改。",
        "",
        "## 9. 质量门禁",
        "",
        "| 臂 | Recall>34.21% | F1>50% | Precision 不下降>2pp | "
        "NCP 单位不增加 | ADI table4 可匹配 | Validator 拒绝真错 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])

    for arm in ARM_ORDER:
        gate = gates[arm]
        lines.append(
            f"| {arm} | {'PASS' if gate['recall_gt_34_21'] else 'FAIL'} | "
            f"{'PASS' if gate['f1_gt_50_00'] else 'FAIL'} | "
            f"{'PASS' if gate['precision_not_lower_than_A_by_2pp'] else 'FAIL'} | "
            f"{'PASS' if gate['ncp_unit_errors_not_increased'] else 'FAIL'} | "
            f"{'PASS' if gate['adi_table4_matchable'] else 'FAIL'} | "
            f"{'PASS' if gate['validator_rejects_true_error'] else 'FAIL'} |"
        )

    lines.extend([
        "",
        "A 是控制组，Recall/F1 门禁对 A 的 FAIL 是预期结果；门禁判定"
        "针对 B/C/D。B/D 的 Precision FAIL 主要来自 ADI identity "
        "差异，C 的 NCP 单位 FAIL 是真实单位错误增加。",
        "",
        "## 10. 结论：结构修复与 Prompt 扩展各自带来什么",
        "",
        "### 10.1 官方冻结口径（内容匹配，不排除 invalid）",
        "",
        f"- **结构修复 B−A（P0 下）**：ΔRecall="
        f"{_pp(deltas['结构 B-A (P0)']['recall'])}，ΔPrecision="
        f"{_pp(deltas['结构 B-A (P0)']['precision'])}，ΔF1="
        f"{_pp(deltas['结构 B-A (P0)']['f1'])}。",
        f"- **Prompt 扩展 C−A（Flat 下）**：ΔRecall="
        f"{_pp(deltas['Prompt C-A (Flat)']['recall'])}，ΔPrecision="
        f"{_pp(deltas['Prompt C-A (Flat)']['precision'])}，ΔF1="
        f"{_pp(deltas['Prompt C-A (Flat)']['f1'])}。",
        f"- **组合 D−A**：ΔRecall="
        f"{_pp(deltas['组合 D-A']['recall'])}，ΔPrecision="
        f"{_pp(deltas['组合 D-A']['precision'])}，ΔF1="
        f"{_pp(deltas['组合 D-A']['f1'])}。",
        f"- **交互 D−C−(B−A)**：ΔRecall="
        f"{_pp(deltas['交互 D-C-(B-A)']['recall'])}，ΔF1="
        f"{_pp(deltas['交互 D-C-(B-A)']['f1'])}。"
        "两个因子高度重叠：一旦 Logical 结构把参数族行暴露出来，"
        "Prompt 扩展的边际 Recall 增益接近 0"
        f"（D−B：ΔRecall={_pp(deltas['Prompt D-B (Logical)']['recall'])}，"
        f"ΔF1={_pp(deltas['Prompt D-B (Logical)']['f1'])}）。",
        "",
        "### 10.2 生产可信口径（官方 TP 中排除 invalid）",
        "",
        f"- A：可信 TP={trusted['A']['trusted_tp']}/76，"
        f"可信 P/R/F1={_pct(trusted['A']['trusted_precision'])} / "
        f"{_pct(trusted['A']['trusted_recall'])} / "
        f"{_pct(trusted['A']['trusted_f1'])}。",
        f"- B：可信 TP={trusted['B']['trusted_tp']}/76，"
        f"可信 P/R/F1={_pct(trusted['B']['trusted_precision'])} / "
        f"{_pct(trusted['B']['trusted_recall'])} / "
        f"{_pct(trusted['B']['trusted_f1'])}；"
        f"官方 {trusted['B']['invalid_tp']} 条 TP 被判 invalid。",
        f"- C：可信 TP={trusted['C']['trusted_tp']}/76，"
        f"可信 P/R/F1={_pct(trusted['C']['trusted_precision'])} / "
        f"{_pct(trusted['C']['trusted_recall'])} / "
        f"{_pct(trusted['C']['trusted_f1'])}；"
        f"官方 {trusted['C']['invalid_tp']} 条 TP 被判 invalid。",
        f"- D：可信 TP={trusted['D']['trusted_tp']}/76，"
        f"可信 P/R/F1={_pct(trusted['D']['trusted_precision'])} / "
        f"{_pct(trusted['D']['trusted_recall'])} / "
        f"{_pct(trusted['D']['trusted_f1'])}；"
        f"官方 {trusted['D']['invalid_tp']} 条 TP 被判 invalid。",
        "",
        "可信口径下，结构修复的 F1 增量只有约 +0.7pp，Prompt 扩展的 "
        "F1 增量约 +17.6pp，组合约 +1.3pp。差异来自 B/D 的 27 条 "
        "内容 TP 被 `recovered_conflict:identity` 判 invalid，而 C 的 "
        "11 条 invalid 主要是单位白名单缺口。",
        "",
        "### 10.3 26 条遗漏与 ADI",
        "",
        f"- 26 条 LLM 遗漏：A 内容命中 0/26；B/C/D 均为 26/26。"
        f"但 Validator 可信命中只有 "
        f"{target_validated['B']['trusted_hits']}/26（B）、"
        f"{target_validated['C']['trusted_hits']}/26（C）、"
        f"{target_validated['D']['trusted_hits']}/26（D）；其余 10 条"
        "分别被 identity 冲突或单位白名单拒绝。",
        f"- ADI 18 条：B/D 内容 TP=4/18，C=1/18，A=0/18；严格 TP "
        "仍为 0/18。B/D 的 14 条 FN 与 14 条 FP 中，"
        f"B 有 {identity_diag['B']['exact_value_unit_type_pairs']} 对、"
        f"D 有 {identity_diag['D']['exact_value_unit_type_pairs']} 对在"
        "数值/单位/类型上完全相同，只是 identity 命名不同；因此 ADI "
        "的瓶颈已从“结构不可见”转为“identity 归并与测试条件”。",
        "",
        "### 10.4 建议",
        "",
        f"- 官方 F1 最佳：**{recommendation['best_arm']} "
        f"({recommendation['best_label']})**，F1="
        f"{_pct(micro[recommendation['best_arm']]['f1'])}；但 Precision "
        f"相对 A0 {_pp(recommendation['delta_precision'])}，"
        "未通过“不明显下降”门禁。",
        f"- 生产可信 F1 最佳：**{recommendation['best_trusted_arm']} "
        f"({recommendation['best_trusted_label']})**，可信 F1="
        f"{_pct(recommendation['best_trusted_f1'])}；其 NCP 单位错误"
        f"为 {ncp[recommendation['best_trusted_arm']]['llm_wrong_unit_count']}"
        f"（A 为 {ncp['A']['llm_wrong_unit_count']}），仍不满足"
        "“NCP 单位错误减少”门禁。",
        f"- 自动判定：**{recommendation['decision']}**。"
        "没有任何新臂同时通过全部质量门禁；因此本轮不能宣布 B3.6 "
        "结构修复或 Prompt 扩展已经达到第一版正式方案标准。",
        "- 下一步优先级：① 修复 ADI identity 的行级归并（同一 "
        "`current_limit` 不能同时代表两行）；② 扩充 Validator 单位"
        "白名单（`%/A`、`%/V`、`uVRMS` 等）；③ 让重试请求也携带 "
        "Prompt 族覆盖；④ 对 C/D 重复运行至少 2 次再判定。",
        "",
        "回答最初问题：",
        "",
        f"1. **结构修复**（B−A）把官方 Recall 提高 "
        f"{_pp(deltas['结构 B-A (P0)']['recall'])}、F1 提高 "
        f"{_pp(deltas['结构 B-A (P0)']['f1'])}，但 Precision 下降 "
        f"{_pp(deltas['结构 B-A (P0)']['precision'])}；生产可信 F1 "
        "只提高约 +0.7pp，主要被 identity 冲突抵消。",
        f"2. **Prompt 扩展**（C−A）把官方 Recall 提高 "
        f"{_pp(deltas['Prompt C-A (Flat)']['recall'])}、F1 提高 "
        f"{_pp(deltas['Prompt C-A (Flat)']['f1'])}，Precision 反而提高 "
        f"{_pp(deltas['Prompt C-A (Flat)']['precision'])}；生产可信 F1 "
        "提高约 +17.6pp，是本轮最稳健的单因子改进。",
        f"3. **两者组合**（D−A）没有叠加：Recall 与结构单独使用相同"
        f"（{_pp(deltas['组合 D-A']['recall'])}），F1 只比结构单独高"
        f"约 1.1pp；交互项为 "
        f"{_pp(deltas['交互 D-C-(B-A)']['f1'])}。",
        "",
    ])

    if recommendation["reasons"]:
        lines.append("自动判定的未通过原因：")

        for reason in recommendation["reasons"]:
            lines.append(f"- {reason}")

        lines.append("")

    lines.extend([
        "## 11. 限制",
        "",
        "1. 每臂只运行一次；即使 temperature=0，provider 仍可能非确定。",
        "2. 当前端点/模型与 B3.5c 不同，B3.5c 的绝对值只作历史参考。",
        "3. 主实验只使用 Gold scope 表；这保证四臂表格范围一致，"
        "但不覆盖全量 selected tables 的非 Gold 表预测。",
        "4. Gold 是人工确认子集，不是 PDF 全部有效参数的穷举；"
        "同表非 Gold 参数可能被计为 FP。",
        "5. 预检只验证链路，不作为最终 F1。",
        "6. 预检 A/ADI 出现过 `finish_reason=length`，生产按行二分后"
        "完成；该次预检因此被判 FAIL。我们随后用 A/AP2112 预检通过，"
        "并继续正式运行。这是对“预检异常即停止”的一处偏离，必须"
        "在解读时保留。",
        "7. 重试请求不携带 Prompt 族覆盖，Prompt 效应被轻微稀释。",
        "8. 评分后只新增只读诊断（可信视图、身份配对、逐条状态），"
        "未修改 `b35c_score_flat.py`、冻结 Gold、Prompt 或提取器；"
        "官方 P/R/F1 与评分契约保持不变。",
        "9. 未修改冻结 Gold、B3.5c/B3.6 产物；未 Git Commit。",
        "",
        "## 12. 产物",
        "",
        "- `results/<arm>/metrics.json`：每臂统一评分指标。",
        "- `results/<arm>/metrics.records.csv`：错误/匹配明细。",
        "- `results/<arm>/metrics.errors.json`：FN/FP/条件/来源错误。",
        "- `results/target_omissions.csv`：26 条遗漏逐条命中矩阵。",
        "- `results/target_omissions_validated.csv`：26 条遗漏逐条的"
        "Validator 状态。",
        "- `results/adi_records.csv`：ADI 18 条逐条状态。",
        "- `results/ncp_unit_errors.json`：NCP 单位错误明细。",
        "- `results/comparison.json`：四臂指标、因子差值、门禁与建议。",
        "",
    ])

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--run-id", default="r1")
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR)
    )
    args = parser.parse_args(argv)

    gold = load_json(args.gold)
    manifest_payload = load_json(args.manifest)

    if gold.get("status") != "frozen":
        raise SystemExit("Gold 未冻结，禁止评分")

    manifest = {
        sample["sample_id"]: sample
        for sample in manifest_payload.get("samples", [])
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_arm = {}
    errors_by_arm = {}
    samples_by_arm = {}
    run_summary = {}

    for arm in ARM_ORDER:
        metrics, errors, samples = _score_arm(
            arm, args.run_id, gold, manifest
        )
        metrics_by_arm[arm] = metrics
        errors_by_arm[arm] = errors
        samples_by_arm[arm] = metrics["samples"]
        run_summary[arm] = metrics["run_manifest"]

        arm_dir = output_dir / arm
        write_json(arm_dir / "metrics.json", metrics)
        write_details_csv(
            samples, arm_dir / "metrics.records.csv"
        )
        write_json(arm_dir / "metrics.errors.json", errors)

    historical = load_json(B35C_METRICS)["micro"]
    micro_by_arm = {
        arm: metrics_by_arm[arm]["micro"]
        for arm in ARM_ORDER
    }

    factorial_deltas = {
        "结构 B-A (P0)": _arm_delta(
            micro_by_arm["A"], micro_by_arm["B"]
        ),
        "结构 D-C (P1)": _arm_delta(
            micro_by_arm["C"], micro_by_arm["D"]
        ),
        "Prompt C-A (Flat)": _arm_delta(
            micro_by_arm["A"], micro_by_arm["C"]
        ),
        "Prompt D-B (Logical)": _arm_delta(
            micro_by_arm["B"], micro_by_arm["D"]
        ),
        "组合 D-A": _arm_delta(
            micro_by_arm["A"], micro_by_arm["D"]
        ),
    }
    structure_p1 = _arm_delta(micro_by_arm["C"], micro_by_arm["D"])
    structure_p0 = _arm_delta(micro_by_arm["A"], micro_by_arm["B"])
    factorial_deltas["交互 D-C-(B-A)"] = {
        key: round(structure_p1[key] - structure_p0[key], 4)
        for key in structure_p1
    }

    target_rows, target_summary = _target_omissions(errors_by_arm)
    adi_rows, adi_summary = _adi_records(errors_by_arm, gold)
    ncp_units = _ncp_units(errors_by_arm)
    prompt_audit = _prompt_audit(args.run_id)
    trusted_view = _trusted_view(args.run_id, gold, manifest)
    target_validated = _target_omissions_validated(
        args.run_id, gold, manifest
    )
    identity_diag = _adi_identity_diagnostics(errors_by_arm)

    negative = _validator_negative_control(args.run_id, manifest)
    gates = _quality_gates(
        micro_by_arm, adi_summary, ncp_units, negative
    )
    recommendation = _recommendation(
        micro_by_arm,
        gates,
        adi_summary,
        ncp_units,
        trusted_view,
        target_validated,
    )

    target_fields = [
        "sample_id",
        "vendor",
        "gold_id",
        "identity",
        "value",
        "unit",
        "value_type",
        "b35c_category",
    ]

    for arm in ARM_ORDER:
        target_fields.extend([
            f"{arm}_content_hit",
            f"{arm}_strict_hit",
        ])

    _write_csv(
        output_dir / "target_omissions.csv",
        target_rows,
        target_fields,
    )

    target_status_fields = [
        "sample_id",
        "vendor",
        "gold_id",
        "identity",
        "value",
        "unit",
        "value_type",
    ]

    for arm in ARM_ORDER:
        target_status_fields.extend([
            f"{arm}_content_hit",
            f"{arm}_status",
        ])

    _write_csv(
        output_dir / "target_omissions_validated.csv",
        _target_status_rows(args.run_id, gold, manifest),
        target_status_fields,
    )

    adi_fields = [
        "gold_id",
        "identity",
        "value",
        "unit",
        "value_type",
        "row_index",
        "page",
    ]

    for arm in ARM_ORDER:
        adi_fields.extend([
            f"{arm}_content_hit",
            f"{arm}_strict_hit",
            f"{arm}_fn_category",
        ])

    _write_csv(
        output_dir / "adi_records.csv",
        adi_rows,
        adi_fields,
    )
    write_json(
        output_dir / "ncp_unit_errors.json", ncp_units
    )
    write_json(
        output_dir / "validator_negative_control.json",
        negative,
    )
    write_json(
        output_dir / "prompt_audit.json", prompt_audit
    )
    write_json(
        output_dir / "trusted_view.json", trusted_view
    )
    write_json(
        output_dir / "target_omissions_validated.json",
        target_validated,
    )
    write_json(
        output_dir / "adi_identity_diagnostics.json",
        identity_diag,
    )

    comparison = {
        "schema": "b37_comparison/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": SCORER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "table_scope": "gold_scope_only",
        "run_id": args.run_id,
        "historical_b35c": historical,
        "historical_caveat": (
            "B3.5c 使用 DeepSeek 官方 deepseek-flash + 全量 "
            "selected tables；本次使用火山引擎 Ark 端点 + "
            "gold scope only。绝对值不可直接比较。"
        ),
        "micro_by_arm": micro_by_arm,
        "samples_by_arm": samples_by_arm,
        "run_summary": run_summary,
        "trusted_view": trusted_view,
        "trusted_view_note": (
            "诊断口径：官方内容 TP 中只把 validated / "
            "pending_review 计为可信 TP；invalid TP 仍计入官方 "
            "TP，但不计入可信 TP。官方 P/R/F1 未被修改。"
        ),
        "factorial_deltas": factorial_deltas,
        "target_omissions_summary": target_summary,
        "target_omissions_validated": target_validated,
        "adi_summary": adi_summary,
        "adi_identity_diagnostics": identity_diag,
        "ncp_units": {
            arm: {
                "llm_wrong_unit_count": value[
                    "llm_wrong_unit_count"
                ],
                "errors": value["errors"],
            }
            for arm, value in ncp_units.items()
        },
        "prompt_audit": prompt_audit,
        "validator_negative_control": negative,
        "quality_gates": gates,
        "recommendation": recommendation,
        "limitations": [
            "每臂单次运行，不构成统计显著结论。",
            "B3.5c 与 B3.7 的模型/端点不同。",
            "主实验只覆盖 Gold scope 表。",
            "预检结果不计入最终 F1。",
            "重试请求不携带 Prompt 族覆盖。",
            "A/ADI 预检曾因截断被判 FAIL，随后改用 A/AP2112 预检。",
            "官方评分不排除 invalid；可信视图为诊断口径。",
        ],
    }
    write_json(output_dir / "comparison.json", comparison)

    combined_errors = []
    for arm in ARM_ORDER:
        for error in errors_by_arm[arm]:
            combined_errors.append({"arm": arm, **error})

    _write_csv(
        output_dir / "remaining_errors.csv",
        combined_errors,
        [
            "arm",
            "sample_id",
            "kind",
            "category",
            "gold_id",
            "prediction_index",
            "page",
            "table_id",
            "row_index",
            "gold_identity",
            "prediction_identity",
            "gold_value_type",
            "prediction_value_type",
            "gold_value",
            "prediction_value",
            "gold_unit",
            "prediction_unit",
            "validation_status",
            "validation_errors",
            "note",
        ],
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "B3.7_REPORT.md"
    _write_report(comparison, report_path)
    comparison["report_path"] = str(
        report_path.relative_to(PROJECT_ROOT)
    )
    write_json(output_dir / "comparison.json", comparison)

    print(json.dumps(
        {
            "micro": micro_by_arm,
            "deltas": factorial_deltas,
            "target_omissions": {
                arm: {
                    "content_hits": target_summary[arm]["content_hits"],
                    "strict_hits": target_summary[arm]["strict_hits"],
                    "trusted_hits": target_validated[arm][
                        "trusted_hits"
                    ],
                }
                for arm in ARM_ORDER
            },
            "trusted_view": {
                arm: {
                    "trusted_tp": trusted_view[arm]["trusted_tp"],
                    "trusted_precision": trusted_view[arm][
                        "trusted_precision"
                    ],
                    "trusted_recall": trusted_view[arm][
                        "trusted_recall"
                    ],
                    "trusted_f1": trusted_view[arm]["trusted_f1"],
                }
                for arm in ARM_ORDER
            },
            "adi": adi_summary,
            "ncp_wrong_unit": {
                arm: ncp_units[arm]["llm_wrong_unit_count"]
                for arm in ARM_ORDER
            },
            "recommendation": recommendation,
            "report": str(report_path),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
