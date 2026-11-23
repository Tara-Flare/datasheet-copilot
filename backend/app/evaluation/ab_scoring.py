"""阶段 B3.2b：A/B 参数级评估。

评估原则：
- 同一份 Gold eval 文件同时用于 Flat / Logical；
- 主匹配键为 logical_row_id + value_type；
- 严格匹配要求身份、数值、单位、行级测试条件都正确；
- 表级默认条件和脚注不要求 LLM 逐字输出，单独记录为 Gold 补充；
- 无法可靠映射的预测明确标记为 unmatched，不做猜测。

B3.4c：在评估层增加参数族语义优先级。
生产 Validator / 来源映射保持不变；这里只修正评分器的
identity 归一化，避免 "Output voltage noise" 先被
"output voltage" 规则吞掉。
"""

from collections import defaultdict
from decimal import Decimal, InvalidOperation

from app.validation.recovered_evidence import (
    logical_family,
    original_family,
)


SCORER_VERSION = "b3.4c-v1"
FAMILY_NORMALIZATION_VERSION = "b3.4c-family-v1"

GROUND_CURRENT_IDS = {
    "t9.p8",
    "t9.p9",
    "t9.p10",
    "t9.p11",
    "t9.p12",
    "t9.p13",
}
VDO_IDS = {f"t9.p{i}" for i in range(18, 32)}
UVLO_IDS = {"t9.p34", "t9.p35", "t9.p36"}

UNTRUSTED_MAPPING_STATUSES = {
    "mapped_source_conflict",
    "unknown_logical_row_id",
    "ambiguous_row_index",
    "missing_source",
}


FAMILY_ALIASES = {
    "nominal_accuracy": "accuracy",
    "accuracy_over_temperature": "accuracy",
    "ground_current_vs_load_current": "ground_current_load",
    "ground_current_in_dropout": "ground_current_dropout",
    "output_voltage_noise": "output_noise",
    "load_regulation_2": "load_regulation",
}

# 仅用于评估 identity 的语义符号；
# 不修改生产 Validator 的 family 规则。
NOISE_SYMBOLS = {
    "vn",
    "v_n",
    "vnoise",
    "v_noise",
    "enoise",
}


def _to_decimal(value):
    text = str(value or "").strip()

    text = (
        text.replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
    )

    try:
        return Decimal(text)
    except (InvalidOperation, ValueError, AttributeError):
        return None


def normalize_unit(unit):
    return (
        str(unit or "")
        .strip()
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
    )


def normalize_conditions(text):
    value = (
        str(text or "")
        .replace("–", "-")
        .replace("−", "-")
        .replace("‒", "-")
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("×", "x")
        .replace("|", ";")
    )

    value = " ".join(value.split())
    value = value.replace(" ;", ";").replace("; ", ";")

    return value.lower().strip()


def normalize_family(value):
    text = str(value or "").strip().lower()

    # 去掉脚注编号和尾部标号
    while "(" in text and ")" in text:
        start = text.find("(")
        end = text.find(")", start)
        if end == -1:
            break
        text = text[:start] + text[end + 1:]

    normalized = "".join(
        char if char.isalnum() else "_"
        for char in text
    )
    normalized = "_".join(
        part for part in normalized.split("_") if part
    )

    return FAMILY_ALIASES.get(normalized, normalized)


def _semantic_family(symbol, name):
    """评估层语义优先规则。

    优先级：noise > accuracy > 原有 family 规则。

    这样可以避免 "Output voltage noise" 被
    "output voltage" 前缀规则归为 output_voltage；
    同时不改变生产 Validator 的 logical_family()。
    """

    symbol_text = normalize_family(symbol)
    name_text = normalize_family(name)
    combined = f"{symbol_text} {name_text}".strip()

    has_noise = (
        symbol_text in NOISE_SYMBOLS
        or name_text in NOISE_SYMBOLS
        or (
            "noise" in combined
            and "rejection" not in combined
        )
    )

    if has_noise:
        if (
            "input" in combined
            or symbol_text.startswith("vin")
        ):
            return "input_noise"

        return "output_noise"

    if "accuracy" in combined:
        return "accuracy"

    return None


def gold_family(record):
    symbol = record.get("parameter_symbol")
    name = record.get("parameter_name")
    semantic = _semantic_family(symbol, name)

    if semantic is not None:
        return semantic

    return normalize_family(
        logical_family({
            "parameter_symbol": symbol,
            "parameter_name": name,
        })
    )


def prediction_family(name):
    semantic = _semantic_family(None, name)

    if semantic is not None:
        return semantic

    return normalize_family(original_family(name))


def _map_by_evidence_row(gold_records, evidence):
    table_id = evidence.get("table_id")
    row_index = evidence.get("row_index")

    primary = {
        item["logical_row_id"]
        for item in gold_records
        if item.get("table_id") == table_id
        and item.get("primary_row_index") == row_index
    }

    if len(primary) == 1:
        return next(iter(primary)), "primary_row_index"

    source = {
        item["logical_row_id"]
        for item in gold_records
        if item.get("table_id") == table_id
        and row_index in (
            item.get("source_row_indices") or []
        )
    }

    if len(source) == 1:
        return next(iter(source)), "source_row_index"

    return None, "unmapped"


def _features(prediction, gold):
    identity = (
        prediction_family(prediction.get("name"))
        == gold_family(gold)
    )
    pred_value = _to_decimal(prediction.get("value"))
    gold_value = _to_decimal(gold.get("value"))

    value = (
        pred_value is not None
        and gold_value is not None
        and pred_value == gold_value
    )
    unit = (
        normalize_unit(prediction.get("unit"))
        == normalize_unit(gold.get("unit"))
    )
    pred_conditions = str(
        prediction.get("test_conditions") or ""
    ).strip()
    gold_conditions = str(
        gold.get("test_conditions_row_level")
        or gold.get("test_conditions")
        or ""
    ).strip()
    conditions_exact = (
        pred_conditions == gold_conditions
    )
    conditions = (
        normalize_conditions(pred_conditions)
        == normalize_conditions(gold_conditions)
    )

    return {
        "identity": identity,
        "value": value,
        "unit": unit,
        "value_type": (
            prediction.get("value_type")
            == gold.get("value_type")
        ),
        "test_conditions_exact": conditions_exact,
        "test_conditions": conditions,
        "predicted_value": prediction.get("value"),
        "gold_value": gold.get("value"),
    }


def _quality(features):
    return (
        int(all((
            features["identity"],
            features["value"],
            features["unit"],
            features["test_conditions"],
        ))),
        int(features["identity"]),
        int(features["value"]),
        int(features["unit"]),
        int(features["test_conditions"]),
        int(features["test_conditions_exact"]),
    )


def _metric(numerator, denominator):
    if denominator == 0:
        return None

    return round(numerator / denominator, 4)


def _f1(precision, recall):
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall
        / (precision + recall),
        4,
    )


def _source_check(
    prediction,
    gold,
    recovered,
):
    evidence = prediction.get("evidence") or {}
    logical_row_id = evidence.get("logical_row_id")
    expected_id = gold["logical_row_id"]

    if logical_row_id == expected_id:
        return {
            "correct": True,
            "method": "logical_row_id",
            "logical_row_id_present": True,
            "recovered_status": (
                recovered or {}
            ).get("recovered_status"),
        }

    if recovered is not None:
        selected = recovered.get(
            "selected_logical_row_id"
        )
        source = (
            recovered.get("field_checks") or {}
        ).get("source") or {}

        if (
            selected == expected_id
            and source.get("status") == "match"
        ):
            return {
                "correct": True,
                "method": "recovered_selected_row",
                "logical_row_id_present": False,
                "recovered_status": recovered.get(
                    "recovered_status"
                ),
            }

    return {
        "correct": False,
        "method": "unverified",
        "logical_row_id_present": (
            logical_row_id == expected_id
        ),
        "recovered_status": (
            recovered or {}
        ).get("recovered_status"),
    }


def _group_metrics(gold_details, ids):
    selected = [
        item for item in gold_details
        if item["logical_row_id"] in ids
    ]
    total = len(selected)
    matched = sum(
        1 for item in selected
        if item["match_status"] == "matched"
    )
    strict = sum(
        1 for item in selected
        if item["strict_match"]
    )
    missing = sum(
        1 for item in selected
        if item["match_status"] == "missing"
    )

    return {
        "logical_ids": sorted(ids),
        "gold_records": total,
        "matched_keys": matched,
        "strict_matched": strict,
        "missing": missing,
        "key_recall": _metric(matched, total),
        "strict_recall": _metric(strict, total),
    }


def score_extraction(
    gold,
    extraction,
    recovered_report=None,
):
    gold_records = gold["expected_parameters"]
    gold_by_key = {
        (
            item["logical_row_id"],
            item["value_type"],
        ): item
        for item in gold_records
    }
    gold_ids = {
        item["logical_row_id"]
        for item in gold_records
    }
    recovered_by_index = {}

    if recovered_report is not None:
        recovered_by_index = {
            index: item
            for index, item in enumerate(
                recovered_report.get("parameters", [])
            )
        }

    predictions = extraction.get("parameters", [])
    candidates = defaultdict(list)
    false_positives = []

    for index, prediction in enumerate(predictions):
        evidence = prediction.get("evidence") or {}
        logical_row_id = evidence.get("logical_row_id")
        map_method = "logical_row_id"

        if logical_row_id not in gold_ids:
            recovered = recovered_by_index.get(index)
            selected = (
                recovered or {}
            ).get("selected_logical_row_id")
            mapping_status = prediction.get(
                "source_mapping_status"
            )

            if (
                selected in gold_ids
                and mapping_status
                not in UNTRUSTED_MAPPING_STATUSES
            ):
                logical_row_id = selected
                map_method = "recovered_selected_row"
            else:
                logical_row_id, map_method = (
                    _map_by_evidence_row(
                        gold_records, evidence
                    )
                )

        key = (
            logical_row_id,
            prediction.get("value_type"),
        )

        if logical_row_id is None or key not in gold_by_key:
            false_positives.append({
                "prediction_index": index,
                "name": prediction.get("name"),
                "value": prediction.get("value"),
                "unit": prediction.get("unit"),
                "value_type": prediction.get(
                    "value_type"
                ),
                "test_conditions": prediction.get(
                    "test_conditions"
                ),
                "evidence": evidence,
                "reason": (
                    "unmapped_source"
                    if logical_row_id is None
                    else "not_in_gold"
                ),
                "map_method": map_method,
            })
            continue

        gold_record = gold_by_key[key]
        features = _features(
            prediction, gold_record
        )
        candidates[key].append({
            "prediction_index": index,
            "prediction": prediction,
            "gold": gold_record,
            "features": features,
            "map_method": map_method,
            "quality": _quality(features),
        })

    matched_by_key = {}

    for key, items in candidates.items():
        items.sort(
            key=lambda item: item["quality"],
            reverse=True,
        )
        matched_by_key[key] = items[0]

        for duplicate in items[1:]:
            false_positives.append({
                "prediction_index": duplicate[
                    "prediction_index"
                ],
                "name": duplicate["prediction"].get(
                    "name"
                ),
                "value": duplicate["prediction"].get(
                    "value"
                ),
                "unit": duplicate["prediction"].get(
                    "unit"
                ),
                "value_type": duplicate[
                    "prediction"
                ].get("value_type"),
                "test_conditions": duplicate[
                    "prediction"
                ].get("test_conditions"),
                "evidence": duplicate[
                    "prediction"
                ].get("evidence"),
                "reason": "duplicate_key",
                "map_method": duplicate[
                    "map_method"
                ],
            })

    details = []

    for gold_record in gold_records:
        key = (
            gold_record["logical_row_id"],
            gold_record["value_type"],
        )
        match = matched_by_key.get(key)
        supplement = gold_record.get(
            "supplement"
        ) or {}
        base = {
            "logical_row_id": key[0],
            "value_type": key[1],
            "parameter_name": gold_record.get(
                "parameter_name"
            ),
            "parameter_symbol": gold_record.get(
                "parameter_symbol"
            ),
            "gold_value": gold_record.get("value"),
            "unit": gold_record.get("unit"),
            "test_conditions_row_level": gold_record.get(
                "test_conditions_row_level"
            ) or gold_record.get("test_conditions"),
            "gold_footnotes": list(
                supplement.get("footnotes") or []
            ),
            "gold_supplemental_condition_changes": list(
                supplement.get("condition_changes") or []
            ),
        }

        if match is None:
            base.update({
                "match_status": "missing",
                "strict_match": False,
                "fields": None,
                "predicted": None,
                "evidence_check": None,
            })
            details.append(base)
            continue

        prediction = match["prediction"]
        recovered = recovered_by_index.get(
            match["prediction_index"]
        )
        source_check = _source_check(
            prediction,
            gold_record,
            recovered,
        )
        features = match["features"]
        strict = all((
            features["identity"],
            features["value"],
            features["unit"],
            features["value_type"],
            features["test_conditions"],
        ))

        base.update({
            "match_status": "matched",
            "strict_match": strict,
            "fields": features,
            "predicted": {
                "index": match["prediction_index"],
                "name": prediction.get("name"),
                "value": prediction.get("value"),
                "unit": prediction.get("unit"),
                "value_type": prediction.get(
                    "value_type"
                ),
                "test_conditions": prediction.get(
                    "test_conditions"
                ),
                "evidence": prediction.get("evidence"),
                "map_method": match["map_method"],
            },
            "evidence_check": source_check,
        })
        details.append(base)

    predicted_count = len(predictions)
    matched_keys = sum(
        1 for item in details
        if item["match_status"] == "matched"
    )
    strict_matched = sum(
        1 for item in details if item["strict_match"]
    )
    gold_count = len(gold_records)

    precision = _metric(
        strict_matched, predicted_count
    )
    recall = _metric(strict_matched, gold_count)
    key_precision = _metric(
        matched_keys, predicted_count
    )
    key_recall = _metric(matched_keys, gold_count)

    def field_accuracy(field):
        return _metric(
            sum(
                1 for item in details
                if item["match_status"] == "matched"
                and item["fields"][field]
            ),
            matched_keys,
        )

    evidence_correct = sum(
        1 for item in details
        if item["match_status"] == "matched"
        and item["evidence_check"]["correct"]
    )
    logical_id_present = sum(
        1 for item in details
        if item["match_status"] == "matched"
        and item["evidence_check"][
            "logical_row_id_present"
        ]
    )

    field_mismatches = []

    for item in details:
        if item["match_status"] != "matched":
            continue

        failed = [
            field
            for field in (
                "identity",
                "value",
                "unit",
                "value_type",
                "test_conditions",
            )
            if not item["fields"][field]
        ]

        if failed:
            field_mismatches.append({
                "logical_row_id": item[
                    "logical_row_id"
                ],
                "value_type": item["value_type"],
                "fields": failed,
                "gold": {
                    "value": item["gold_value"],
                    "unit": item["unit"],
                    "conditions": item[
                        "test_conditions_row_level"
                    ],
                },
                "predicted": item["predicted"],
            })

    return {
        "scorer_version": SCORER_VERSION,
        "family_normalization_version": (
            FAMILY_NORMALIZATION_VERSION
        ),
        "gold_count": gold_count,
        "predicted_count": predicted_count,
        "matched_keys": matched_keys,
        "strict_matched": strict_matched,
        "missing_count": gold_count - matched_keys,
        "false_positive_count": len(false_positives),
        "duplicate_count": sum(
            1 for item in false_positives
            if item["reason"] == "duplicate_key"
        ),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "key_precision": key_precision,
        "key_recall": key_recall,
        "key_f1": _f1(key_precision, key_recall),
        "field_accuracy": {
            "identity": field_accuracy("identity"),
            "min_typ_max": field_accuracy(
                "value_type"
            ),
            "value": field_accuracy("value"),
            "unit": field_accuracy("unit"),
            "test_conditions": field_accuracy(
                "test_conditions"
            ),
            "test_conditions_exact": _metric(
                sum(
                    1 for item in details
                    if item["match_status"] == "matched"
                    and item["fields"][
                        "test_conditions_exact"
                    ]
                ),
                matched_keys,
            ),
        },
        "evidence_source": {
            "correct": evidence_correct,
            "checked_matched": matched_keys,
            "accuracy_matched": _metric(
                evidence_correct, matched_keys
            ),
            "accuracy_all_predictions": _metric(
                evidence_correct, predicted_count
            ),
            "logical_row_id_present": logical_id_present,
            "logical_row_id_presence_rate": _metric(
                logical_id_present, predicted_count
            ),
        },
        "special": {
            "ground_current": _group_metrics(
                details, GROUND_CURRENT_IDS
            ),
            "vdo": _group_metrics(details, VDO_IDS),
            "uvlo": _group_metrics(details, UVLO_IDS),
        },
        "gold_details": details,
        "false_positives": false_positives,
        "field_mismatches": field_mismatches,
        "missing": [
            {
                "logical_row_id": item[
                    "logical_row_id"
                ],
                "value_type": item["value_type"],
                "parameter_name": item[
                    "parameter_name"
                ],
                "gold_value": item["gold_value"],
                "unit": item["unit"],
                "test_conditions": item[
                    "test_conditions_row_level"
                ],
            }
            for item in details
            if item["match_status"] == "missing"
        ],
    }
